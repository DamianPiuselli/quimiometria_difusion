"""Experiment 01: Physical Simulation Sandbox Benchmark.

Compares PLS Baseline, Deterministic 1D-CNN, and In-Context Diffusion Calibration (ICDC)
under realistic analytical conditions:
- EMG chromatographic/spectroscopic peaks
- Heteroscedastic Horwitz noise
- Matrix interference & baseline drift
- Low-concentration non-negativity boundary (LOD/LOQ)
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from icdc.data.synthetic import generate_synthetic_batch
from icdc.models.pls_baseline import PLSBaseline
from icdc.models.cnn_baseline import CNN1DBaseline
from icdc.models.diffusion import DiffusionRegressor
from icdc.metrology.metrics import (
    calc_rmsep,
    calc_r2,
    calc_bias,
    calc_picp,
    calc_mpiw,
    MetrologicalReport,
)


def run_synthetic_sandbox(
    n_train: int = 250,
    n_test: int = 80,
    n_points: int = 128,
    output_dir: str = "reports/exp01_synthetic_sandbox/figures",
    random_state: int = 42,
):
    print("=" * 70)
    print("  EXPERIMENT 01: PHYSICAL SIMULATION SANDBOX (ICDC vs PLS vs 1D-CNN)")
    print("=" * 70)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Generate Synthetic Data
    print(f"\n[1/4] Generating synthetic datasets (N_train={n_train}, N_test={n_test}, Points={n_points})...")
    x_train, y_train, gains_train = generate_synthetic_batch(
        n_samples=n_train,
        n_points=n_points,
        conc_range=(0.02, 10.0),
        matrix_prob=0.35,
        matrix_max_ratio=0.4,
        gain_std=0.04,
        random_state=random_state,
    )
    x_test, y_test, gains_test = generate_synthetic_batch(
        n_samples=n_test,
        n_points=n_points,
        conc_range=(0.02, 10.0),
        matrix_prob=0.35,
        matrix_max_ratio=0.4,
        gain_std=0.04,
        random_state=random_state + 100,
    )

    # 2. Train Models
    print("\n[2/4] Training Baseline Models & Diffusion Regressor...")

    # A. PLS (sin SNV en cromatografía para no normalizar el área del pico)
    print("  --> Fitting PLS Baseline (use_snv=False for chromatography)...")
    pls = PLSBaseline(max_components=10, use_snv=False, random_state=random_state)
    pls.fit(x_train, y_train)
    print(f"      Selected Latent Variables (LV): {pls.best_n_components}")

    # B. 1D-CNN
    print("  --> Training 1D-CNN Baseline (50 epochs)...")
    cnn = CNN1DBaseline(feature_dim=32, lr=1e-3, epochs=60, batch_size=32, random_state=random_state)
    cnn.fit(x_train, y_train)

    # C. Diffusion Regressor (ICDC)
    print("  --> Training Diffusion Regressor (Score matching, 60 epochs)...")
    diffusion = DiffusionRegressor(
        spectral_dim=32,
        context_dim=16,
        timesteps=50,
        lr=1e-3,
        epochs=70,
        batch_size=32,
        random_state=random_state,
    )
    diffusion.fit(x_train, y_train)

    # 3. Evaluate Predictions and Intervals
    print("\n[3/4] Evaluating Predictions and 95% Confidence / Credible Intervals...")
    # PLS Predictions
    y_pred_pls, pls_low, pls_up = pls.predict_intervals(x_test, coverage=0.95)
    pls_rep = MetrologicalReport(
        model_name="PLS Baseline",
        n_samples=n_test,
        rmsep=calc_rmsep(y_test, y_pred_pls),
        r2=calc_r2(y_test, y_pred_pls),
        bias=calc_bias(y_test, y_pred_pls),
        picp_95=calc_picp(y_test, pls_low, pls_up),
        mpiw_95=calc_mpiw(pls_low, pls_up),
        non_negative_fraction=float(np.mean(pls_low >= 0.0)),
    )

    # CNN Predictions
    y_pred_cnn, cnn_low, cnn_up = cnn.predict_intervals(x_test, coverage=0.95)
    cnn_rep = MetrologicalReport(
        model_name="1D-CNN Baseline",
        n_samples=n_test,
        rmsep=calc_rmsep(y_test, y_pred_cnn),
        r2=calc_r2(y_test, y_pred_cnn),
        bias=calc_bias(y_test, y_pred_cnn),
        picp_95=calc_picp(y_test, cnn_low, cnn_up),
        mpiw_95=calc_mpiw(cnn_low, cnn_up),
        non_negative_fraction=float(np.mean(cnn_low >= 0.0)),
    )

    # Diffusion Predictions (50 reverse diffusion steps, 80 Monte Carlo samples per test point)
    print("  --> Sampling from Diffusion posterior distribution (Monte Carlo N=80)...")
    y_pred_diff, diff_low, diff_up = diffusion.predict_intervals(x_test, coverage=0.95, n_samples=80)
    diff_rep = MetrologicalReport(
        model_name="ICDC Diffusion",
        n_samples=n_test,
        rmsep=calc_rmsep(y_test, y_pred_diff),
        r2=calc_r2(y_test, y_pred_diff),
        bias=calc_bias(y_test, y_pred_diff),
        picp_95=calc_picp(y_test, diff_low, diff_up),
        mpiw_95=calc_mpiw(diff_low, diff_up),
        non_negative_fraction=float(np.mean(diff_low >= 0.0)),
    )

    # Print Summary Table
    print("\n" + "=" * 70)
    print(f"{'Modelo':<20} | {'RMSEP':<8} | {'R²':<8} | {'PICP (95%)':<12} | {'MPIW':<8} | {'y_min >= 0':<10}")
    print("-" * 70)
    for rep in [pls_rep, cnn_rep, diff_rep]:
        print(
            f"{rep.model_name:<20} | {rep.rmsep:<8.4f} | {rep.r2:<8.4f} | "
            f"{(rep.picp_95 * 100 if rep.picp_95 else 0.0):<11.1f}% | "
            f"{rep.mpiw_95:<8.4f} | {(rep.non_negative_fraction * 100):<9.1f}%"
        )
    print("=" * 70)

    # 4. Generate Diagnostic Figures
    print("\n[4/4] Generating Diagnostic Figures...")
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    models_data = [
        ("PLS Baseline", y_pred_pls, pls_low, pls_up, "#1f77b4", axes[0]),
        ("1D-CNN Baseline", y_pred_cnn, cnn_low, cnn_up, "#ff7f0e", axes[1]),
        ("ICDC (Diffusion)", y_pred_diff, diff_low, diff_up, "#2ca02c", axes[2]),
    ]

    for title, y_p, low, up, color, ax in models_data:
        # Ideal line
        ax.plot([0, 10.5], [0, 10.5], "k--", alpha=0.5, label="Ideal (y=x)")
        # Error bars
        y_err_low = np.maximum(0, y_p - low)
        y_err_up = np.maximum(0, up - y_p)
        ax.errorbar(
            y_test,
            y_p,
            yerr=[y_err_low, y_err_up],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2,
            alpha=0.75,
            markersize=5,
            label="Predicción ± U95%",
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Concentración Real ($y$)", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel(r"Concentración Predicha ($\hat{y}$)", fontsize=11)
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-1.5, 12.0)
        ax.legend(loc="upper left")

    plt.tight_layout()
    fig_path = out_path / "01_calibration_curves_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  --> Saved calibration comparison to: {fig_path}")

    # Figure 2: Uncertainty Width vs Concentration (Heteroscedasticity test)
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, pls_up - pls_low, label="PLS (Homocedástico)", alpha=0.7, color="#1f77b4")
    plt.scatter(y_test, diff_up - diff_low, label=r"ICDC Difusión (Heterocedástico)", alpha=0.7, color="#2ca02c")
    plt.title(r"Amplitud del Intervalo de Incertidumbre ($U_{95\%}$) vs. Concentración", fontsize=12, fontweight="bold")
    plt.xlabel("Concentración Real ($y$)", fontsize=11)
    plt.ylabel("Amplitud del Intervalo ($y_{upper} - y_{lower}$)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    fig2_path = out_path / "02_uncertainty_vs_concentration.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"  --> Saved uncertainty analysis to: {fig2_path}")

    print("\nExperiment 01 finished successfully.")
    return pls_rep, cnn_rep, diff_rep


if __name__ == "__main__":
    run_synthetic_sandbox()
