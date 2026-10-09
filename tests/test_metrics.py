import numpy as np
from icdc.metrology.metrics import (
    calc_rmsep,
    calc_bias,
    calc_r2,
    calc_picp,
    calc_mpiw,
    horwitz_sd,
    calc_horrat,
    MetrologicalReport,
)


def test_basic_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.0, 4.2, 4.8])

    rmsep = calc_rmsep(y_true, y_pred)
    assert 0.0 < rmsep < 0.2

    bias = calc_bias(y_true, y_pred)
    assert abs(bias) < 0.1

    r2 = calc_r2(y_true, y_pred)
    assert r2 > 0.98


def test_interval_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_lower = np.array([0.8, 1.8, 2.5, 3.7])
    y_upper = np.array([1.2, 2.2, 3.5, 4.3])

    picp = calc_picp(y_true, y_lower, y_upper)
    assert picp == 1.0

    mpiw = calc_mpiw(y_lower, y_upper)
    assert np.isclose(mpiw, 0.60)


def test_horwitz():
    # 1 ppm = 1e-6 mass fraction
    h_sd = horwitz_sd(1e-6)
    assert h_sd > 0
    # Relative SD (RSD) at 1 ppm should be ~16%
    rsd = (h_sd / 1e-6) * 100.0
    assert 10.0 < rsd < 25.0

    horrat = calc_horrat(empirical_sd=h_sd * 1.2, concentration_mass_fraction=1e-6)
    assert np.isclose(horrat, 1.2)


def test_report():
    rep = MetrologicalReport(
        model_name="PLS",
        n_samples=50,
        rmsep=0.12,
        r2=0.99,
        bias=0.01,
        picp_95=0.94,
        mpiw_95=0.45,
    )
    d = rep.to_dict()
    assert d["model_name"] == "PLS"
    assert "94.0%" in str(rep)
