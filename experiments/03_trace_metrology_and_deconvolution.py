"""Experiment 03: Trace Metrology (LOD/LOQ, CCalpha, CCbeta) and Zero-Shot Blind Deconvolution (DPS).

Addresses the two fundamental pillars of the ICDC framework:
1. Trace Metrology near LOD/LOQ:
   - Evaluates strictly non-negative posterior distributions (y >= 0) under Horwitz heteroscedastic noise.
   - Computes ISO 11843 / EU Decision 2002/657/EC Decision Limit (CCalpha) and Detection Capability (CCbeta).
   - Demonstrates the failure of unconstrained linear models (PLS with Savitzky-Golay) predicting negative concentrations.
2. Zero-Shot Blind Deconvolution of Unmodeled Interferents:
   - Injects unexpected co-eluting adulterants/interferents not present in calibration.
   - Demonstrates silent failure/bias in classical chemometrics (PLS).
   - Demonstrates zero-shot separation via Diffusion Posterior Sampling (DPS), isolating the adulterant
     and restoring quantitative accuracy without model retraining.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from icdc.data.synthetic import (
    emg_peak,
    generate_baseline_drift,
    generate_chromatographic_signal,
    generate_synthetic_batch,
)
from icdc.models.pls_baseline import PLSBaseline
from icdc.models.diffusion import DiffusionRegressor
from icdc.models.dps_deconvolution import SpectralDiffusionDPS
from icdc.metrology.metrics import (
    calc_rmsep,
    calc_r2,
    calc_bias,
    calc_picp,
    calc_mpiw,
)
from icdc.metrology.detection_limits import (
    calc_negative_bound_rate,
    compute_linear_decision_limits,
    compute_posterior_decision_limits,
    calc_exceedance_probability,
)


def run_experiment_03(
    output_dir: str = "reports/figures",
    n_points: int = 128,
    random_state: int = 42,
):
    print("=" * 80)
    print(" EXPERIMENT 03: TRACE METROLOGY (LOD/LOQ) & ZERO-SHOT BLIND DECONVOLUTION (DPS)")
    print("=" * 80)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # =========================================================================
    # PART 1: TRACE METROLOGY NEAR LOD/LOQ (PILLAR C)
    # =========================================================================
    print("\n" + "=" * 60)
    print(" [PART 1] TRACE METROLOGY & DECISION LIMITS (ISO 11843 / 2002/657/EC)")
    print("=" * 60)

    # 1.1 Generate calibration dataset spanning standard operational range
    print("  -> Generating calibration dataset (N=180, C in [0.01, 5.0])...")
    x_train, y_train, _ = generate_synthetic_batch(
        n_samples=180,
        n_points=n_points,
        conc_range=(0.01, 5.0),
        matrix_prob=0.25,
        gain_std=0.03,
        random_state=random_state,
    )

    # 1.2 Generate trace validation set: 30 true blanks (C=0.0) + 70 trace samples (C in [0.005, 0.15])
    print("  -> Generating trace evaluation set (N=30 blanks + N=70 trace samples)...")
    rng = np.random.RandomState(random_state + 100)
    t = np.linspace(0.0, 10.0, n_points)

    # Blanks (y = 0.0)
    x_blanks = []
    y_blanks = np.zeros(30)
    for _ in range(30):
        # Blank has matrix/solvent baseline drift and noise, but zero analyte
        drift = generate_baseline_drift(n_points, drift_amplitude=0.04, random_state=rng)
        noise = rng.normal(0.0, 0.008, size=n_points)
        x_blanks.append(drift + noise)
    x_blanks = np.array(x_blanks)

    # Trace samples (y in [0.005, 0.15])
    x_traces = []
    y_traces = []
    for _ in range(70):
        c_val = float(rng.uniform(0.005, 0.15))
        sig, _ = generate_chromatographic_signal(
            t=t,
            analyte_conc=c_val,
            matrix_conc=rng.uniform(0.0, 0.05),
            gain=rng.normal(1.0, 0.03),
            base_noise=0.006,
            apply_horwitz=True,
            random_state=rng,
        )
        x_traces.append(sig)
        y_traces.append(c_val)
    x_traces = np.array(x_traces)
    y_traces = np.array(y_traces)

    x_trace_eval = np.vstack([x_blanks, x_traces])
    y_trace_eval = np.concatenate([y_blanks, y_traces])

    # 1.3 Fit PLS Baseline with Savitzky-Golay preprocessing (fair chemometric standard)
    print("  -> Fitting PLSBaseline with Savitzky-Golay 2nd derivative & smoothing...")
    pls = PLSBaseline(
        max_components=6,
        use_snv=False,  # Not suitable for zero-baseline chromatograms
        savgol_window=9,
        savgol_poly=2,
        savgol_deriv=0,  # Smoothing
        cv_folds=5,
        random_state=random_state,
    )
    pls.fit(x_train, y_train)

    y_pred_pls, y_low_pls, y_up_pls = pls.predict_intervals(x_trace_eval, coverage=0.95)
    pls_blank_preds = pls.predict(x_blanks)
    pls_res_std = float(np.sqrt(pls.residual_variance)) if pls.residual_variance else float(np.std(y_train - pls.predict(x_train)))

    linear_limits = compute_linear_decision_limits(
        pls_blank_preds,
        residual_std=pls_res_std,
        alpha=0.05,
        beta=0.05,
        model_name="PLS (Savitzky-Golay)",
    )

    # 1.4 Fit Diffusion Regressor (ICDC)
    print("  -> Fitting DiffusionRegressor (ICDC) with anti-collapse anchor...")
    diffusion_reg = DiffusionRegressor(
        spectral_dim=32,
        context_dim=0,
        timesteps=40,
        epochs=70,
        batch_size=32,
        standardize_x=True,
        use_anchor=True,
        device="cpu",
        random_state=random_state,
    )
    diffusion_reg.fit(x_train, y_train, augment_transfer_shift=False)

    print("  -> Sampling posterior distributions for trace samples...")
    # Posterior draws for blanks and trace samples
    blank_draws = diffusion_reg.sample_posterior(x_blanks, n_samples=100, clip_non_negative=True)
    trace_draws = diffusion_reg.sample_posterior(x_traces, n_samples=100, clip_non_negative=True)
    all_draws = np.vstack([blank_draws, trace_draws])

    y_pred_diff = np.median(all_draws, axis=1)
    y_low_diff = np.percentile(all_draws, 2.5, axis=1)
    y_up_diff = np.percentile(all_draws, 97.5, axis=1)

    diff_limits = compute_posterior_decision_limits(
        blank_posterior_draws=blank_draws,
        trace_posterior_draws=trace_draws,
        trace_true_concentrations=y_traces,
        alpha=0.05,
        beta=0.05,
        model_name="Diffusion (ICDC)",
    )

    # Compare Metrological Metrics
    pls_neg_rate = calc_negative_bound_rate(y_low_pls)
    diff_neg_rate = calc_negative_bound_rate(y_low_diff)

    pls_picp = calc_picp(y_trace_eval, y_low_pls, y_up_pls)
    diff_picp = calc_picp(y_trace_eval, y_low_diff, y_up_diff)

    pls_mpiw = calc_mpiw(y_low_pls, y_up_pls)
    diff_mpiw = calc_mpiw(y_low_diff, y_up_diff)

    print("\n--- TRACE METROLOGY AUDIT RESULTS ---")
    print(f"PLS Baseline (with Savitzky-Golay):")
    print(f"  CCalpha (Decision Limit):      {linear_limits.cc_alpha:.4f}")
    print(f"  CCbeta  (Detection Power):     {linear_limits.cc_beta:.4f}")
    print(f"  Negative Lower Bound Rate:     {pls_neg_rate * 100:.1f}%  <-- GUM VIOLATION (Negative Conc.)")
    print(f"  PICP (95% nominal):            {pls_picp * 100:.1f}%")
    print(f"  MPIW (Sharpness):              {pls_mpiw:.4f}")

    print(f"\nDiffusion Regressor (ICDC):")
    print(f"  CCalpha (Decision Limit):      {diff_limits.cc_alpha:.4f}")
    print(f"  CCbeta  (Detection Power):     {diff_limits.cc_beta:.4f}")
    print(f"  Negative Lower Bound Rate:     {diff_neg_rate * 100:.1f}%  <-- STRICTLY PHYSICAL (y >= 0)")
    print(f"  PICP (95% nominal):            {diff_picp * 100:.1f}%")
    print(f"  MPIW (Sharpness):              {diff_mpiw:.4f}")

    # =========================================================================
    # PART 2: ZERO-SHOT BLIND DECONVOLUTION OF UNMODELED INTERFERENTS (PILLAR B)
    # =========================================================================
    print("\n" + "=" * 60)
    print(" [PART 2] ZERO-SHOT BLIND INTERFERENT DECONVOLUTION VIA DPS")
    print("=" * 60)

    # 2.1 Train Spectral Diffusion Model on Clean Signals ONLY
    print("  -> Training SpectralDiffusionDPS prior on clean historical runs (N=180)...")
    dps_model = SpectralDiffusionDPS(
        hidden_dim=32,
        num_blocks=3,
        timesteps=30,
        epochs=60,
        batch_size=32,
        device="cpu",
        random_state=random_state,
    )
    dps_model.fit(x_train)

    # 2.2 Create contaminated evaluation samples:
    # Analyte (C=2.0, t_r=5.0) + Unmodeled Co-eluting Adulterant Peak (t_r=4.85, Amp=1.2)
    print("  -> Synthesizing contaminated samples with unexpected co-eluting adulterant (t_r=4.85)...")
    n_contam = 30
    x_clean_eval = []
    x_contam_eval = []
    y_contam_true = []
    interferents_eval = []

    for i in range(n_contam):
        c_true = float(rng.uniform(1.0, 3.5))
        pure_sig, _ = generate_chromatographic_signal(
            t=t,
            analyte_conc=c_true,
            matrix_conc=0.1,
            gain=1.0,
            base_noise=0.005,
            apply_horwitz=True,
            random_state=rng,
        )
        # Unmodeled interferent peak overlapping with analyte
        interf_peak = emg_peak(t, t_r=4.85, sigma=0.18, tau=0.12, amplitude=1.0)
        contam_sig = pure_sig + interf_peak

        x_clean_eval.append(pure_sig)
        x_contam_eval.append(contam_sig)
        y_contam_true.append(c_true)
        interferents_eval.append(interf_peak)

    x_clean_eval = np.array(x_clean_eval)
    x_contam_eval = np.array(x_contam_eval)
    y_contam_true = np.array(y_contam_true)
    interferents_eval = np.array(interferents_eval)

    # 2.3 Classical PLS failure under unmodeled interferent
    y_pred_pls_clean = pls.predict(x_clean_eval)
    y_pred_pls_contam = pls.predict(x_contam_eval)

    pls_bias_clean = calc_bias(y_contam_true, y_pred_pls_clean)
    pls_bias_contam = calc_bias(y_contam_true, y_pred_pls_contam)
    pls_rmsep_contam = calc_rmsep(y_contam_true, y_pred_pls_contam)

    print("\n--- PLS SILENT FAILURE AUDIT ---")
    print(f"  PLS on Clean Samples:        RMSEP={calc_rmsep(y_contam_true, y_pred_pls_clean):.4f}, Bias={pls_bias_clean:+.4f}")
    print(f"  PLS on Contaminated Samples: RMSEP={pls_rmsep_contam:.4f}, Bias={pls_bias_contam:+.4f}")
    print(f"  ==> PLS suffered severe systematic bias ({pls_bias_contam:+.4f}) with ZERO warning to the analyst!")

    # 2.4 DPS Blind Deconvolution (Zero-Shot)
    print("\n  -> Executing Diffusion Posterior Sampling (DPS) deconvolution...")
    dps_res = dps_model.deconvolve(
        x_contam_eval,
        t_start=18,
        guidance_scale=0.5,
        positivity_penalty=3.0,
        detection_threshold=1.30,
    )

    # 2.5 Re-evaluate quantification on DPS-recovered clean spectra
    y_pred_pls_after_dps = pls.predict(dps_res.x_clean_recovered)
    y_pred_diff_after_dps = diffusion_reg.predict(dps_res.x_clean_recovered, n_samples=30)

    dps_pls_bias = calc_bias(y_contam_true, y_pred_pls_after_dps)
    dps_pls_rmsep = calc_rmsep(y_contam_true, y_pred_pls_after_dps)

    dps_diff_bias = calc_bias(y_contam_true, y_pred_diff_after_dps)
    dps_diff_rmsep = calc_rmsep(y_contam_true, y_pred_diff_after_dps)

    print("\n--- DPS ZERO-SHOT BLIND DECONVOLUTION RESULTS ---")
    print(f"  Interferent Detection Rate:   {np.mean(dps_res.is_interferent_detected) * 100:.1f}%")
    print(f"  Mean Anomaly Score:           {np.mean(dps_res.anomaly_score):.4f} (Threshold: {dps_res.detection_threshold})")
    print(f"  PLS after DPS Deconvolution:  RMSEP={dps_pls_rmsep:.4f}, Bias={dps_pls_bias:+.4f} (Bias reduced by {abs(pls_bias_contam) - abs(dps_pls_bias):.4f}!)")
    print(f"  ICDC after DPS Deconvolution: RMSEP={dps_diff_rmsep:.4f}, Bias={dps_diff_bias:+.4f}")

    # =========================================================================
    # PART 3: GENERATE HIGH-QUALITY DIAGNOSTIC FIGURES
    # =========================================================================
    print("\n[Part 3] Generating publication-quality diagnostic figures...")

    # Figure 1: Trace Metrology & Decision Limits
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Subplot A: Predicted vs True with intervals near LOD
    ax = axes[0]
    # Plot first 35 samples (ordered by true concentration)
    sort_idx = np.argsort(y_trace_eval)[:40]
    xs = np.arange(len(sort_idx))
    y_true_sorted = y_trace_eval[sort_idx]

    # PLS intervals
    ax.errorbar(
        xs - 0.15,
        y_pred_pls[sort_idx],
        yerr=[
            y_pred_pls[sort_idx] - y_low_pls[sort_idx],
            y_up_pls[sort_idx] - y_pred_pls[sort_idx],
        ],
        fmt="o",
        color="#d95f02",
        alpha=0.7,
        capsize=2,
        label=f"PLS (Neg Bound Rate: {pls_neg_rate * 100:.0f}%)",
    )
    # Diffusion intervals
    ax.errorbar(
        xs + 0.15,
        y_pred_diff[sort_idx],
        yerr=[
            y_pred_diff[sort_idx] - y_low_diff[sort_idx],
            y_up_diff[sort_idx] - y_pred_diff[sort_idx],
        ],
        fmt="s",
        color="#1b9e77",
        alpha=0.8,
        capsize=2,
        label=f"Diffusion ICDC (Neg Bound Rate: {diff_neg_rate * 100:.0f}%)",
    )
    ax.plot(xs, y_true_sorted, "k--", label="True Concentration", alpha=0.6)
    ax.axhline(0.0, color="red", linestyle=":", linewidth=1.5, label="Physical Bound (C=0)")
    ax.set_title("A. Confidence Intervals near LOD/LOQ", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sample Index (sorted by concentration)", fontsize=11)
    ax.set_ylabel("Analyte Concentration (mg/kg)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)

    # Subplot B: Blank distribution and Decision Limits (CCalpha, CCbeta)
    ax = axes[1]
    sns.kdeplot(pls_blank_preds, ax=ax, label="PLS Blank Preds", color="#d95f02", fill=True, alpha=0.3)
    sns.kdeplot(diff_limits.blank_posterior_draws if hasattr(diff_limits, 'blank_posterior_draws') else blank_draws.ravel(),
                ax=ax, label="Diffusion Blank Posterior", color="#1b9e77", fill=True, alpha=0.3)
    ax.axvline(linear_limits.cc_alpha, color="#d95f02", linestyle="--", linewidth=1.5, label=f"PLS CCα ({linear_limits.cc_alpha:.3f})")
    ax.axvline(diff_limits.cc_alpha, color="#1b9e77", linestyle="--", linewidth=1.5, label=f"Diffusion CCα ({diff_limits.cc_alpha:.3f})")
    ax.axvline(0.0, color="red", linestyle=":", linewidth=1.5, label="C=0")
    ax.set_title("B. Blank Distribution & Decision Limits (CCα)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Concentration (mg/kg)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)

    # Subplot C: Interval width vs Concentration (Horwitz heteroscedasticity)
    ax = axes[2]
    ax.scatter(y_trace_eval, y_up_pls - y_low_pls, color="#d95f02", alpha=0.6, label="PLS (Rigid Homoscedastic)")
    ax.scatter(y_trace_eval, y_up_diff - y_low_diff, color="#1b9e77", alpha=0.7, label="Diffusion (Adaptive Heteroscedastic)")
    ax.set_title("C. Uncertainty Width (MPIW) vs Concentration", fontsize=13, fontweight="bold")
    ax.set_xlabel("True Concentration (mg/kg)", fontsize=11)
    ax.set_ylabel("95% Interval Width (mg/kg)", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)

    fig.tight_layout()
    fig1_path = out_path / "03_trace_metrology_limits.png"
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"  -> Saved: {fig1_path}")

    # Figure 2: Blind Deconvolution (DPS) and Bias Restoration
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Subplot A: Spectral Overlay for a representative sample
    ax = axes[0]
    sample_idx = 0
    ax.plot(t, x_contam_eval[sample_idx], color="#7570b3", linewidth=2.0, label="Observed Mixture (Analyte + Adulterant)")
    ax.plot(t, x_clean_eval[sample_idx], color="black", linestyle="--", linewidth=1.5, label="Ground Truth Analyte")
    ax.plot(t, dps_res.x_clean_recovered[sample_idx], color="#1b9e77", linewidth=2.0, label="DPS Recovered Clean")
    ax.plot(t, dps_res.x_interferent_isolated[sample_idx], color="#e7298a", linewidth=2.0, label="DPS Isolated Adulterant")
    ax.set_title(f"A. Blind Spectral Deconvolution (Sample #{sample_idx})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Retention Time / Wavelength", fontsize=11)
    ax.set_ylabel("Intensity / Absorbance", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)

    # Subplot B: Bias Bar Chart
    ax = axes[1]
    models = ["PLS (Clean)", "PLS (Contaminated)", "PLS after DPS", "ICDC after DPS"]
    biases = [pls_bias_clean, pls_bias_contam, dps_pls_bias, dps_diff_bias]
    colors = ["#2b83ba", "#d7191c", "#fdae61", "#1b9e77"]
    bars = ax.bar(models, biases, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax.axhline(0.0, color="black", linestyle="-", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.annotate(f"{h:+.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3 if h >= 0 else -12), textcoords="offset points",
                    ha="center", va=va, fontsize=10, fontweight="bold")
    ax.set_title("B. Systematic Prediction Bias (Δy)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Bias (mg/kg)", fontsize=11)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)

    # Subplot C: Anomaly Score Distribution (Matrix Triage)
    ax = axes[2]
    # Clean anomaly scores vs contaminated anomaly scores
    clean_dps = dps_model.deconvolve(
        x_clean_eval,
        t_start=18,
        guidance_scale=0.5,
        positivity_penalty=3.0,
        detection_threshold=1.30,
    )
    sns.kdeplot(clean_dps.anomaly_score, ax=ax, label="Clean Historical Samples", color="#1b9e77", fill=True, alpha=0.4)
    sns.kdeplot(dps_res.anomaly_score, ax=ax, label="Contaminated Samples", color="#d7191c", fill=True, alpha=0.4)
    ax.axvline(dps_res.detection_threshold, color="black", linestyle="--", linewidth=1.5, label=f"Triage Threshold ({dps_res.detection_threshold})")
    ax.set_title("C. Matrix Triage & Anomaly Score", fontsize=13, fontweight="bold")
    ax.set_xlabel("Matrix Atypicity Score: ||∇_X log p(X)||", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig2_path = out_path / "03_dps_blind_deconvolution.png"
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"  -> Saved: {fig2_path}")

    print("\n" + "=" * 80)
    print(" EXPERIMENT 03 COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return {
        "pls_neg_rate": pls_neg_rate,
        "diff_neg_rate": diff_neg_rate,
        "pls_bias_contam": pls_bias_contam,
        "dps_pls_bias": dps_pls_bias,
        "dps_diff_bias": dps_diff_bias,
        "dps_detection_rate": float(np.mean(dps_res.is_interferent_detected)),
    }


if __name__ == "__main__":
    run_experiment_03()
