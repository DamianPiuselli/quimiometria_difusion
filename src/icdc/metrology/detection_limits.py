"""Metrological Decision Limits and Trace Metrology (LOD, LOQ, CCalpha, CCbeta).

Compliant with:
- ISO 11843-1/2: Capability of detection (Critical value x_C, Minimum detectable value x_D).
- European Commission Decision 2002/657/EC: Decision limit (CCalpha) and Detection capability (CCbeta).
- GUM (Guide to the Expression of Uncertainty in Measurement): Non-parametric posterior integration
  and physical non-negativity constraint auditing (y >= 0).
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Sequence
import numpy as np
from scipy import stats


@dataclass
class DetectionLimitsResult:
    """Encapsulates regulatory detection limits and physical boundary metrics."""

    model_name: str
    cc_alpha: float
    cc_beta: float
    lod: float
    loq: float
    negative_bound_rate: float
    alpha: float = 0.05
    beta: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "cc_alpha": self.cc_alpha,
            "cc_beta": self.cc_beta,
            "lod": self.lod,
            "loq": self.loq,
            "negative_bound_rate": self.negative_bound_rate,
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def __str__(self) -> str:
        return (
            f"[{self.model_name} Metrological Decision Limits (alpha={self.alpha}, beta={self.beta})]\n"
            f"  CCalpha (Decision Limit):     {self.cc_alpha:.4f}\n"
            f"  CCbeta  (Detection Power):    {self.cc_beta:.4f}\n"
            f"  LOD (3.3s):                   {self.lod:.4f}\n"
            f"  LOQ (10s):                    {self.loq:.4f}\n"
            f"  Negative Lower Bound Rate:    {self.negative_bound_rate * 100:.1f}% "
            f"({'VIOLATES GUM/PHYSICS' if self.negative_bound_rate > 0 else 'STRICTLY PHYSICAL (y >= 0)'})"
        )


def calc_negative_bound_rate(y_lower: np.ndarray) -> float:
    """Calculate fraction of samples where the lower prediction interval drops below zero.

    In quantitative chemistry, y represents concentration mass fraction (e.g. mg/kg),
    which is strictly non-negative. A negative lower bound represents a physical absurdity
    introduced by unconstrained homoscedastic linear assumptions.
    """
    y_lower = np.asarray(y_lower, dtype=np.float64).ravel()
    if len(y_lower) == 0:
        return 0.0
    return float(np.mean(y_lower < 0.0))


def calc_exceedance_probability(posterior_samples: np.ndarray, threshold: float) -> np.ndarray:
    """Calculate empirical exceedance probability P(y > threshold | X).

    Given posterior sample matrix (n_samples, n_draws), returns array of length n_samples
    with the fraction of draws exceeding the regulatory threshold (e.g. MRL).
    """
    samples = np.asarray(posterior_samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    return np.mean(samples > threshold, axis=1)


def compute_linear_decision_limits(
    y_blank_preds: np.ndarray,
    residual_std: float,
    alpha: float = 0.05,
    beta: float = 0.05,
    model_name: str = "Linear/PLS",
) -> DetectionLimitsResult:
    """Compute parametric ISO 11843 / 2002/657/EC limits assuming Gaussian error.

    CCalpha = mean(y_blank) + z_(1-alpha) * s_blank
    CCbeta  = CCalpha + z_(1-beta) * s_blank
    """
    y_blank_preds = np.asarray(y_blank_preds, dtype=np.float64).ravel()
    blank_mean = float(np.mean(y_blank_preds))
    blank_sd = float(residual_std) if residual_std > 0 else float(np.std(y_blank_preds, ddof=1))
    blank_sd = max(blank_sd, 1e-8)

    z_alpha = float(stats.norm.ppf(1.0 - alpha))
    z_beta = float(stats.norm.ppf(1.0 - beta))

    cc_alpha = blank_mean + z_alpha * blank_sd
    cc_beta = cc_alpha + z_beta * blank_sd

    lod = blank_mean + 3.3 * blank_sd
    loq = blank_mean + 10.0 * blank_sd

    # Margin for 95% CI around blank
    margin = stats.norm.ppf(1.0 - alpha / 2.0) * blank_sd
    y_lower_blanks = y_blank_preds - margin
    neg_rate = calc_negative_bound_rate(y_lower_blanks)

    return DetectionLimitsResult(
        model_name=model_name,
        cc_alpha=cc_alpha,
        cc_beta=cc_beta,
        lod=lod,
        loq=loq,
        negative_bound_rate=neg_rate,
        alpha=alpha,
        beta=beta,
    )


def compute_posterior_decision_limits(
    blank_posterior_draws: np.ndarray,
    trace_posterior_draws: Optional[np.ndarray] = None,
    trace_true_concentrations: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    beta: float = 0.05,
    model_name: str = "Diffusion (ICDC)",
) -> DetectionLimitsResult:
    """Compute empirical non-parametric decision limits from posterior draws p(y | X_blank).

    - CCalpha is the (1 - alpha) quantile of blank posterior predictions.
    - If trace_posterior_draws and trace_true_concentrations are provided:
      CCbeta is determined as the minimum true concentration where P(y_pred > CCalpha) >= 1 - beta.
      Otherwise, CCbeta is estimated via blank posterior dispersion above CCalpha.
    """
    # blank_posterior_draws shape: (n_blanks, n_draws) or flat draws
    blanks = np.asarray(blank_posterior_draws, dtype=np.float64)
    all_blank_draws = blanks.ravel()

    # Empirical (1 - alpha) quantile on blank posterior
    cc_alpha = float(np.percentile(all_blank_draws, (1.0 - alpha) * 100.0))
    blank_median = float(np.median(all_blank_draws))
    blank_mad = float(np.median(np.abs(all_blank_draws - blank_median)))
    blank_sd_equiv = max(1.4826 * blank_mad, 1e-8)

    # CCbeta calculation
    if trace_posterior_draws is not None and trace_true_concentrations is not None:
        trace_draws = np.asarray(trace_posterior_draws, dtype=np.float64)
        c_true = np.asarray(trace_true_concentrations, dtype=np.float64).ravel()
        # Compute detection rate (P(draws > CCalpha)) for each concentration level
        det_rates = np.mean(trace_draws > cc_alpha, axis=1)

        # Find lowest concentration where detection rate >= 1 - beta
        qualified = np.where(det_rates >= (1.0 - beta))[0]
        if len(qualified) > 0:
            cc_beta = float(np.min(c_true[qualified]))
        else:
            # Extrapolate if highest tested concentration didn't reach 1-beta
            z_beta = float(stats.norm.ppf(1.0 - beta))
            cc_beta = cc_alpha + z_beta * blank_sd_equiv
    else:
        z_beta = float(stats.norm.ppf(1.0 - beta))
        cc_beta = cc_alpha + z_beta * blank_sd_equiv

    lod = float(np.percentile(all_blank_draws, 95.0))
    loq = float(np.percentile(all_blank_draws, 99.0)) + 3.0 * blank_sd_equiv

    # Evaluate lower bounds of blank intervals
    alpha_ci = alpha / 2.0
    if blanks.ndim == 2:
        y_lower = np.percentile(blanks, alpha_ci * 100.0, axis=1)
    else:
        y_lower = np.percentile(blanks, alpha_ci * 100.0, axis=0, keepdims=True)

    neg_rate = calc_negative_bound_rate(y_lower)

    return DetectionLimitsResult(
        model_name=model_name,
        cc_alpha=cc_alpha,
        cc_beta=cc_beta,
        lod=lod,
        loq=loq,
        negative_bound_rate=neg_rate,
        alpha=alpha,
        beta=beta,
    )
