"""Experiment 02: Real-World Benchmark on the Corn NIR Dataset.

Tests Cross-Instrument Calibration Transfer and In-Domain Performance:
- Source Instrument: FOSS NIRSystems M5 (Train N=50)
- Target Instrument: FOSS NIRSystems MP5 (Test N=20)
- Transfer Context: N=10 standards from MP5

Compares:
1. In-Domain Performance (M5 -> M5): Proof that models accurately learn without constant collapse.
2. Cross-Instrument Direct Transfer (M5 -> MP5): PLS and 1D-CNN suffering instrument bias.
3. ICDC Diffusion (In-Context Calibration): Transfer adaptation using MP5 standards.
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
    print("=" * 75)
    print("  EXPERIMENT 02: REAL BENCHMARK (CORN NIR M5 -> MP5 CALIBRATION TRANSFER)")
    print("=" * 75)

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

    # In-domain test set on M5
    x_src_test = corn_data["X_m5"][splits["test_idx"]]

    n_test = len(y_tgt_test)
    print(f"  --> Source Training (M5): {len(x_src_train)} samples, {x_src_train.shape[1]} wavelengths.")
    print(f"  --> Transfer Context (MP5): {len(x_tgt_transfer)} standards.")
    print(f"  --> Target Test (MP5): {n_test} unknown samples.")
    print(f"  --> Target Y: Mean={np.mean(y_tgt_test):.3f}, Std={np.std(y_tgt_test):.3f}, Range=[{y_tgt_test.min():.2f}, {y_tgt_test.max():.2f}]")

    # 2. Train Models on Source Instrument (M5)
    print("\n[2/4] Training Models on Source Instrument (M5)...")

    # A. PLS
    print("  --> Fitting PLS on M5 (with cross-validation)...")
    pls = PLSBaseline(max_components=10, use_snv=False, random_state=random_state)
    pls.fit(x_src_train, y_src_train)
    print(f"      Selected Latent Variables: {pls.best_n_components}")

    # B. 1D-CNN (with localized pooling & standardization)
    print("  --> Training 1D-CNN on M5 (100 epochs)...")
    cnn = CNN1DBaseline(feature_dim=32, lr=1e-3, epochs=100, batch_size=16, standardize_x=True, random_state=random_state)
    cnn.fit(x_src_train, y_src_train)

    # C. ICDC Diffusion (with CARD anti-collapse anchor and In-Context context)
    print("  --> Training ICDC Diffusion Model on M5 (120 epochs)...")
    diffusion = DiffusionRegressor(
        spectral_dim=32,
        context_dim=1,
        timesteps=50,
        lr=1e-3,
        epochs=120,
        batch_size=16,
        standardize_x=True,
        use_anchor=True,
        random_state=random_state,
    )
    c_m5_ctx = np.zeros((len(x_src_train), 1), dtype=np.float32)
    diffusion.fit(x_src_train, y_src_train, c_day=c_m5_ctx, augment_transfer_shift=True)

    # 3. In-Domain Verification (M5 -> M5)
    print("\n[3/5] In-Domain Verification (M5 -> M5) to rule out constant collapse:")
    pred_in_pls = pls.predict(x_src_test)
    pred_in_cnn = cnn.predict(x_src_test)
    pred_in_diff = diffusion.predict(x_src_test, n_samples=30, c_day=np.zeros((n_test, 1)))

    print(f"  * PLS (M5 -> M5):      R2 = {calc_r2(y_tgt_test, pred_in_pls):.4f} | RMSEP = {calc_rmsep(y_tgt_test, pred_in_pls):.4f} | Range = [{pred_in_pls.min():.2f}, {pred_in_pls.max():.2f}]")
    print(f"  * 1D-CNN (M5 -> M5):   R2 = {calc_r2(y_tgt_test, pred_in_cnn):.4f} | RMSEP = {calc_rmsep(y_tgt_test, pred_in_cnn):.4f} | Range = [{pred_in_cnn.min():.2f}, {pred_in_cnn.max():.2f}]")
    print(f"  * Diffusion (M5 -> M5): R2 = {calc_r2(y_tgt_test, pred_in_diff):.4f} | RMSEP = {calc_rmsep(y_tgt_test, pred_in_diff):.4f} | Range = [{pred_in_diff.min():.2f}, {pred_in_diff.max():.2f}]")

    # 4. Cross-Instrument Transfer (M5 -> MP5)
    print("\n[4/5] Evaluating Cross-Instrument Transfer on MP5...")

    # PLS Direct (without transfer)
    y_pred_pls, pls_low, pls_up = pls.predict_intervals(x_tgt_test, coverage=0.95)
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

    # ICDC Diffusion (In-Context Calibration with MP5 transfer standards)
    print("  --> In-Context Diffusion Sampling on MP5 (Monte Carlo N=60)...")
    # Estimate the transfer bias from the 10 MP5 transfer standards
    x_transfer_proc = diffusion.scaler.transform(x_tgt_transfer) if diffusion.scaler is not None else x_tgt_transfer
    pred_transfer_anchor = diffusion.anchor_model.predict(x_transfer_proc)
    c_transfer_bias = float(np.mean(pred_transfer_anchor - y_tgt_transfer)) / diffusion.y_std
    c_day_mp5 = np.full((n_test, 1), c_transfer_bias, dtype=np.float32)

    y_pred_diff, diff_low, diff_up = diffusion.predict_intervals(
        x_tgt_test, coverage=0.95, n_samples=60, c_day=c_day_mp5
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
    print("\n" + "=" * 78)
    print(f"{'Modelo':<28} | {'RMSEP':<8} | {'R²':<8} | {'Bias':<8} | {'PICP (95%)':<11} | {'Pred Range':<14}")
    print("-" * 78)
    for rep, preds in [(pls_rep, y_pred_pls), (cnn_rep, y_pred_cnn), (diff_rep, y_pred_diff)]:
        r_str = f"[{preds.min():.2f}, {preds.max():.2f}]"
        print(
            f"{rep.model_name:<28} | {rep.rmsep:<8.4f} | {rep.r2:<8.4f} | {rep.bias:<+8.4f} | "
            f"{(rep.picp_95 * 100 if rep.picp_95 else 0.0):<10.1f}% | {r_str:<14}"
        )
    print("=" * 78)

    # 5. Generate Diagnostic Figures
    print("\n[5/5] Generating Real Benchmark Diagnostic Figures...")
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
                [y_tgt_test.min() - 0.5, y_tgt_test.max() + 0.5], "k--", alpha=0.5, label="Ideal (y=x)")
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
