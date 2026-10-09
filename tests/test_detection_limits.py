import numpy as np
from icdc.metrology.detection_limits import (
    calc_negative_bound_rate,
    calc_exceedance_probability,
    compute_linear_decision_limits,
    compute_posterior_decision_limits,
)


def test_negative_bound_rate():
    # Bounds with some negatives
    bounds = np.array([-0.05, 0.10, -0.01, 0.05, 0.20])
    rate = calc_negative_bound_rate(bounds)
    assert rate == 0.4

    # All positive
    assert calc_negative_bound_rate(np.array([0.0, 0.1, 0.2])) == 0.0


def test_exceedance_probability():
    # 2 samples, 4 posterior draws each
    draws = np.array([
        [0.05, 0.08, 0.12, 0.15],  # 2 > 0.10 -> 0.5
        [0.01, 0.02, 0.03, 0.04],  # 0 > 0.10 -> 0.0
    ])
    probs = calc_exceedance_probability(draws, threshold=0.10)
    np.testing.assert_allclose(probs, [0.5, 0.0])


def test_compute_linear_decision_limits():
    blanks = np.zeros(20)
    res_std = 0.05
    res = compute_linear_decision_limits(blanks, residual_std=res_std, alpha=0.05, beta=0.05)

    assert res.cc_alpha > 0.0
    assert res.cc_beta > res.cc_alpha
    assert res.lod > 0.0
    assert res.loq > res.lod
    # Blanks centered at 0 with CI will produce negative lower bounds
    assert res.negative_bound_rate == 1.0


def test_compute_posterior_decision_limits():
    # Simulated strictly positive posterior draws (e.g. truncated / log-normal near 0)
    rng = np.random.default_rng(42)
    blank_draws = np.abs(rng.normal(loc=0.01, scale=0.005, size=(15, 100)))

    # Trace level draws
    trace_draws = np.abs(rng.normal(loc=0.05, scale=0.01, size=(5, 100)))
    trace_c = np.array([0.01, 0.03, 0.05, 0.07, 0.10])

    res = compute_posterior_decision_limits(
        blank_posterior_draws=blank_draws,
        trace_posterior_draws=trace_draws,
        trace_true_concentrations=trace_c,
        alpha=0.05,
        beta=0.05,
    )

    assert res.cc_alpha > 0.0
    assert res.cc_beta > 0.0
    assert res.negative_bound_rate == 0.0  # Strictly non-negative
