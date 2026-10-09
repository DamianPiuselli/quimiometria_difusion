"""Experiment 02: Real-World Benchmark on the Corn NIR Dataset.

Tests Cross-Instrument Calibration Transfer:
- Source Instrument: FOSS NIRSystems M5
- Target Instrument: FOSS NIRSystems MP5
- Target: Chemical content (Moisture / Matrix composition)

Compares:
1. PLS Baseline (Direct transfer without adaptation vs. Retrained)
2. 1D-CNN Baseline (Direct transfer)
3. ICDC Diffusion (In-Context Calibration with transfer standards)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from icdc.data.datasets import load_corn_dataset, split_calibration_transfer
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


def run_real_benchmark(
    n_train_source: int = 50,
    n_transfer_standards: int = 10,
    output_dir: str = "reports/figures",
    random_state: int = 42,
):
    print("=" * 70)
    print("  EXPERIMENT 02: REAL BENCHMARK (CORN NIR M5 -> MP5 CALIBRATION TRANSFER)")
    print("=" * 70)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Real Dataset
    print("\n[1/4] Loading Cargill Corn NIR dataset (M5 and MP5 spectrometers)...")
    corn_data = load_corn_dataset()
    splits = split_calibration_transfer(
        x_source=corn_data["X_m5"],
        x_target=corn_data["X_mp5"],
        y=corn_data["y"],
        n_train_source=n_train_source,
        n_transfer_standards=n_transfer_standards,
        random_state=random_state,
    )

    x_src_train = splits["X_source_train"]
    y_src_train = splits["y_source_train"]
    x_tgt_transfer = splits["X_target_transfer"]
    y_tgt_transfer = splits["y_target_transfer"]
    x_tgt_test = splits["X_target_test"]
    y_tgt_test = splits["y_target_test"]

    n_test = len(y_tgt_test)
    print(f"  --> Source Training (M5): {len(x_src_train)} samples, {x_src_train.shape[1]} wavelengths.")
    print(f"  --> Transfer Context (MP5): {len(x_tgt_transfer)} standards.")
    print(f"  --> Target Test (MP5): {n_test} unknown samples.")

    # 2. Train Models on Source Instrument (M5)
    print("\n[2/4] Training Models on Source Instrument (M5)...")

    # A. PLS Direct Transfer
    print("  --> Fitting PLS on M5 (with cross-validation)...")
    pls_direct = PLSBaseline(max_components=12, use_snv=True, random_state=random_state)
    pls_direct.fit(x_src_train, y_src_train)
    print(f"      Selected Latent Variables: {pls_direct.best_n_components}")

    # B. 1D-CNN
    print("  --> Training 1D-CNN on M5 (70 epochs)...")
    cnn = CNN1DBaseline(feature_dim=32, lr=1e-3, epochs=70, batch_size=16, random_state=random_state)
    cnn.fit(x_src_train, y_src_train)

    # C. ICDC Diffusion (with In-Context daily context vector)
    print("  --> Training ICDC Diffusion Model on M5 (80 epochs)...")
    # For training on M5, the source context is the mean profile of M5 standards
    c_m5_proto = np.mean(x_src_train[:n_transfer_standards], axis=0, keepdims=True)
    c_m5_ctx = np.repeat(c_m5_proto[:, :16], len(x_src_train), axis=0)

    diffusion = DiffusionRegressor(
        spectral_dim=32,
        context_dim=16,
        timesteps=50,
        lr=1e-3,
        epochs=80,
        batch_size=16,
        random_state=random_state,
    )
    diffusion.fit(x_src_train, y_src_train, c_day=c_m5_ctx)

    # 3. Predict on Target Instrument (MP5)
    print("\n[3/4] Evaluating Transfer Performance on Target Instrument (MP5)...")

    # PLS Direct (without transfer)
    y_pred_pls, pls_low, pls_up = pls_direct.predict_intervals(x_tgt_test, coverage=0.95)
    pls_rep = MetrologicalReport(
        model_name="PLS (M5 -> MP5 Direct)",
        n_samples=n_test,
        rmsep=calc_rmsep(y_tgt_test, y_pred_pls),
        r2=calc_r2(y_tgt_test, y_pred_pls),
        bias=calc_bias(y_tgt_test, y_pred_pls),
        picp_95=calc_picp(y_tgt_test, pls_low, pls_up),
        mpiw_95=calc_mpiw(pls_low, pls_up),
        non_negative_fraction=float(np.mean(pls_low >= 0.0)),
    )

    # 1D-CNN Direct (without transfer)
    y_pred_cnn, cnn_low, cnn_up = cnn.predict_intervals(x_tgt_test, coverage=0.95)
    cnn_rep = MetrologicalReport(
        model_name="1D-CNN (M5 -> MP5 Direct)",
        n_samples=n_test,
        rmsep=calc_rmsep(y_tgt_test, y_pred_cnn),
        r2=calc_r2(y_tgt_test, y_pred_cnn),
        bias=calc_bias(y_tgt_test, y_pred_cnn),
        picp_95=calc_picp(y_tgt_test, cnn_low, cnn_up),
        mpiw_95=calc_mpiw(cnn_low, cnn_up),
        non_negative_fraction=float(np.mean(cnn_low >= 0.0)),
    )

    # ICDC Diffusion (conditioned on the MP5 transfer standards context)
    print("  --> In-Context Diffusion Sampling on MP5 (Monte Carlo N=60)...")
    c_mp5_proto = np.mean(x_tgt_transfer, axis=0, keepdims=True)
    c_mp5_ctx = np.repeat(c_mp5_proto[:, :16], n_test, axis=0)

    y_pred_diff, diff_low, diff_up = diffusion.predict_intervals(
        x_tgt_test, coverage=0.95, n_samples=60, c_day=c_mp5_ctx
    )
    diff_rep = MetrologicalReport(
        model_name="ICDC Diffusion (In-Context)",
        n_samples=n_test,
        rmsep=calc_rmsep(y_tgt_test, y_pred_diff),
        r2=calc_r2(y_tgt_test, y_pred_diff),
        bias=calc_bias(y_tgt_test, y_pred_diff),
        picp_95=calc_picp(y_tgt_test, diff_low, diff_up),
        mpiw_95=calc_mpiw(diff_low, diff_up),
        non_negative_fraction=float(np.mean(diff_low >= 0.0)),
    )

    # Print Summary Table
    print("\n" + "=" * 75)
    print(f"{'Modelo':<28} | {'RMSEP':<8} | {'R²':<8} | {'Bias':<8} | {'PICP (95%)':<11} | {'MPIW':<8}")
    print("-" * 75)
    for rep in [pls_rep, cnn_rep, diff_rep]:
        print(
            f"{rep.model_name:<28} | {rep.rmsep:<8.4f} | {rep.r2:<8.4f} | {rep.bias:<+8.4f} | "
            f"{(rep.picp_95 * 100 if rep.picp_95 else 0.0):<10.1f}% | {rep.mpiw_95:<8.4f}"
        )
    print("=" * 75)

    # 4. Generate Diagnostic Figures
    print("\n[4/4] Generating Real Benchmark Diagnostic Figures...")
    sns.set_theme(style="whitegrid", font="sans-serif")

    # Plot 1: Spectra Comparison M5 vs MP5
    plt.figure(figsize=(10, 4.5))
    wavelengths = np.linspace(1100, 2498, x_src_train.shape[1])
    plt.plot(wavelengths, x_src_train[0], label="Espectrómetro M5 (Muestra #1)", color="#1f77b4", alpha=0.8)
    plt.plot(wavelengths, x_tgt_transfer[0], label="Espectrómetro MP5 (Misma muestra)", color="#d62728", linestyle="--", alpha=0.8)
    plt.title("Comparación de Espectros NIR Reales: Deriva Inter-Instrumento (M5 vs. MP5)", fontsize=12, fontweight="bold")
    plt.xlabel(r"Longitud de onda ($\text{nm}$)", fontsize=11)
    plt.ylabel(r"Absorbancia ($\log(1/R)$)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    fig1_path = out_path / "03_corn_spectra_drift_comparison.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"  --> Saved spectral drift comparison to: {fig1_path}")

    # Plot 2: Prediction vs Target for the three models
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    models_data = [
        ("PLS (M5 -> MP5 Direct)", y_pred_pls, pls_low, pls_up, "#1f77b4", axes[0]),
        ("1D-CNN (M5 -> MP5 Direct)", y_pred_cnn, cnn_low, cnn_up, "#ff7f0e", axes[1]),
        ("ICDC Diffusion (In-Context)", y_pred_diff, diff_low, diff_up, "#2ca02c", axes[2]),
    ]

    for title, y_p, low, up, color, ax in models_data:
        ax.plot([y_tgt_test.min() - 0.5, y_tgt_test.max() + 0.5],
                [y_tgt_test.min() - 0.5, y_tgt_test.max() + 0.5], "k--", alpha=0.5, label="Ideal")
        y_err_low = np.maximum(0, y_p - low)
        y_err_up = np.maximum(0, up - y_p)
        ax.errorbar(
            y_tgt_test,
            y_p,
            yerr=[y_err_low, y_err_up],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2,
            alpha=0.8,
            markersize=6,
            label=r"Predicción $\pm U_{95\%}$",
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Contenido Real (MP5)", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel(r"Contenido Predicho ($\hat{y}$)", fontsize=11)
        ax.legend(loc="upper left")

    plt.tight_layout()
    fig2_path = out_path / "04_real_corn_transfer_benchmark.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"  --> Saved calibration transfer comparison to: {fig2_path}")

    print("\nExperiment 02 finished successfully.")
    return pls_rep, cnn_rep, diff_rep


if __name__ == "__main__":
    run_real_benchmark()
