"""Metrological evaluation metrics for chemometric and quantitative analytical models.

Compliant with GUM (Guide to the Expression of Uncertainty in Measurement) and
analytical chemistry validation guidelines (ICH Q2(R1), Horwitz model).
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


def calc_rmsep(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error of Prediction (RMSEP)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calc_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean prediction bias: mean(y_pred - y_true)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean(y_pred - y_true))


def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R^2)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def calc_picp(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability (PICP).

    Measures the fraction of true values that fall within the predicted
    credible or confidence intervals [y_lower, y_upper].
    For a 95% nominal interval, ideal PICP is ~0.95.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_lower = np.asarray(y_lower, dtype=np.float64).ravel()
    y_upper = np.asarray(y_upper, dtype=np.float64).ravel()
    covered = (y_true >= y_lower) & (y_true <= y_upper)
    return float(np.mean(covered))


def calc_mpiw(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Mean Prediction Interval Width (MPIW).

    Quantifies the sharpness of the prediction intervals: mean(y_upper - y_lower).
    Narrower intervals with high coverage indicate superior uncertainty calibration.
    """
    y_lower = np.asarray(y_lower, dtype=np.float64).ravel()
    y_upper = np.asarray(y_upper, dtype=np.float64).ravel()
    return float(np.mean(np.maximum(0.0, y_upper - y_lower)))


def horwitz_sd(concentration_mass_fraction: np.ndarray | float) -> np.ndarray | float:
    """Theoretical Horwitz standard deviation for inter-laboratory reproducibility.

    sigma = 0.02 * C^0.8492, where C is dimensionless mass fraction (e.g. 1 ppm = 1e-6).
    """
    c = np.asarray(concentration_mass_fraction, dtype=np.float64)
    c_safe = np.clip(c, 1e-12, 1.0)
    res = 0.02 * (c_safe ** 0.8492)
    if np.ndim(concentration_mass_fraction) == 0:
        return float(res)
    return res


def calc_horrat(empirical_sd: float, concentration_mass_fraction: float) -> float:
    """Horwitz Ratio (HorRat = empirical_sd / horwitz_sd).

    HorRat between 0.5 and 2.0 indicates acceptable analytical performance.
    """
    h_sd = horwitz_sd(concentration_mass_fraction)
    if h_sd <= 0:
        return 0.0
    return float(empirical_sd / h_sd)


@dataclass
class MetrologicalReport:
    """Structured report comparing accuracy and uncertainty metrics."""
    model_name: str
    n_samples: int
    rmsep: float
    r2: float
    bias: float
    picp_95: Optional[float] = None
    mpiw_95: Optional[float] = None
    non_negative_fraction: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "n_samples": self.n_samples,
            "rmsep": self.rmsep,
            "r2": self.r2,
            "bias": self.bias,
            "picp_95": self.picp_95,
            "mpiw_95": self.mpiw_95,
            "non_negative_fraction": self.non_negative_fraction,
        }

    def __str__(self) -> str:
        picp_str = f"{self.picp_95 * 100:.1f}%" if self.picp_95 is not None else "N/A"
        mpiw_str = f"{self.mpiw_95:.4f}" if self.mpiw_95 is not None else "N/A"
        return (
            f"[{self.model_name}] (N={self.n_samples})\n"
            f"  RMSEP: {self.rmsep:.4f} | R²: {self.r2:.4f} | Bias: {self.bias:+.4f}\n"
            f"  PICP (95% target): {picp_str} | MPIW: {mpiw_str}\n"
            f"  Non-negative predictions: {self.non_negative_fraction * 100:.1f}%"
        )
