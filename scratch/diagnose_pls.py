"""Scratch script: Rigorous diagnosis and validation of PLS implementation.

Compares:
1. PLSBaseline vs. pure scikit-learn PLSRegression.
2. The impact of SNV preprocessing on chromatographic vs. spectroscopic data.
3. In-domain (M5 -> M5) vs. Cross-instrument (M5 -> MP5) behavior on Corn NIR.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from icdc.models.pls_baseline import PLSBaseline, apply_snv
from icdc.data.synthetic import generate_synthetic_batch
from icdc.data.datasets import load_corn_dataset, split_calibration_transfer
from icdc.metrology.metrics import calc_rmsep, calc_r2, calc_bias, calc_picp, calc_mpiw


def test_synthetic():
    print("=" * 70)
    print("1. EVALUATION ON SYNTHETIC CHROMATOGRAPHIC DATA")
    print("=" * 70)
    x_train, y_train, _ = generate_synthetic_batch(n_samples=250, n_points=128, conc_range=(0.02, 10.0), random_state=42)
    x_test, y_test, _ = generate_synthetic_batch(n_samples=80, n_points=128, conc_range=(0.02, 10.0), random_state=142)

    # A. Custom PLSBaseline with use_snv=True
    pls_snv = PLSBaseline(max_components=10, use_snv=True, random_state=42).fit(x_train, y_train)
    y_pred_snv, low_snv, up_snv = pls_snv.predict_intervals(x_test)
    print(f"PLSBaseline (with SNV):    LV={pls_snv.best_n_components:2d} | R2={calc_r2(y_test, y_pred_snv):.4f} | RMSEP={calc_rmsep(y_test, y_pred_snv):.4f}")

    # B. Custom PLSBaseline with use_snv=False
    pls_raw = PLSBaseline(max_components=10, use_snv=False, random_state=42).fit(x_train, y_train)
    y_pred_raw, low_raw, up_raw = pls_raw.predict_intervals(x_test)
    print(f"PLSBaseline (without SNV): LV={pls_raw.best_n_components:2d} | R2={calc_r2(y_test, y_pred_raw):.4f} | RMSEP={calc_rmsep(y_test, y_pred_raw):.4f} | Non-neg={np.mean(low_raw >= 0)*100:.1f}%")

    # C. Pure scikit-learn PLSRegression (same number of components)
    sk_pls = PLSRegression(n_components=pls_raw.best_n_components).fit(x_train, y_train)
    y_pred_sk = sk_pls.predict(x_test).ravel()
    print(f"Pure sklearn (LV={pls_raw.best_n_components}, no SNV):  | R2={calc_r2(y_test, y_pred_sk):.4f} | RMSEP={calc_rmsep(y_test, y_pred_sk):.4f}")

    diff = np.max(np.abs(y_pred_raw - y_pred_sk))
    print(f"--> Max absolute discrepancy between PLSBaseline and pure sklearn: {diff:.2e}")
    assert diff < 1e-6, "PLSBaseline differs from pure scikit-learn!"


def test_corn():
    print("\n" + "=" * 70)
    print("2. EVALUATION ON REAL CORN NIR BENCHMARK")
    print("=" * 70)
    corn = load_corn_dataset()
    splits = split_calibration_transfer(corn["X_m5"], corn["X_mp5"], corn["y"], n_train_source=50, n_transfer_standards=10, random_state=42)

    X_src_tr = splits["X_source_train"]
    y_src_tr = splits["y_source_train"]
    X_src_te = corn["X_m5"][splits["test_idx"]]
    X_tgt_te = splits["X_target_test"]
    y_tgt_te = splits["y_target_test"]

    print(f"Dataset stats: N_train={len(X_src_tr)}, N_test={len(X_tgt_te)}, Wavelengths={X_src_tr.shape[1]}")
    print(f"Target Y stats: Mean={np.mean(y_tgt_te):.3f}, Std={np.std(y_tgt_te):.3f}, Var={np.var(y_tgt_te):.4f}")

    # A. In-domain test (M5 -> M5)
    print("\n--- IN-DOMAIN (Train M5 -> Test M5) ---")
    pls_in = PLSBaseline(max_components=12, use_snv=False, random_state=42).fit(X_src_tr, y_src_tr)
    y_pred_in = pls_in.predict(X_src_te)
    print(f"PLS (M5 -> M5):  LV={pls_in.best_n_components:2d} | R2={calc_r2(y_tgt_te, y_pred_in):.4f} | RMSEP={calc_rmsep(y_tgt_te, y_pred_in):.4f} | Bias={calc_bias(y_tgt_te, y_pred_in):+.4f}")

    sk_in = PLSRegression(n_components=pls_in.best_n_components).fit(X_src_tr, y_src_tr)
    y_pred_sk_in = sk_in.predict(X_src_te).ravel()
    print(f"Pure sklearn:   LV={pls_in.best_n_components:2d} | R2={calc_r2(y_tgt_te, y_pred_sk_in):.4f} | RMSEP={calc_rmsep(y_tgt_te, y_pred_sk_in):.4f}")
    assert np.max(np.abs(y_pred_in - y_pred_sk_in)) < 1e-6

    # B. Cross-instrument test (M5 -> MP5)
    print("\n--- CROSS-INSTRUMENT DIRECT TRANSFER (Train M5 -> Test MP5) ---")
    y_pred_cross = pls_in.predict(X_tgt_te)
    bias = calc_bias(y_tgt_te, y_pred_cross)
    rmsep = calc_rmsep(y_tgt_te, y_pred_cross)
    r2 = calc_r2(y_tgt_te, y_pred_cross)
    print(f"PLS (M5 -> MP5): LV={pls_in.best_n_components:2d} | R2={r2:.4f} | RMSEP={rmsep:.4f} | Bias={bias:+.4f}")
    print(f"Mathematical explanation of negative R2:")
    print(f"  MSE = {rmsep**2:.4f}, Var(Y) = {np.var(y_tgt_te):.4f}")
    print(f"  R2 = 1 - (MSE / Var(Y)) = 1 - ({rmsep**2:.4f} / {np.var(y_tgt_te):.4f}) = {1 - (rmsep**2 / np.var(y_tgt_te)):.4f}")


if __name__ == "__main__":
    test_synthetic()
    test_corn()
