"""Synthetic physical simulation of chromatography and spectroscopy signals."""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.special import erfc


def emg_peak(
    t: np.ndarray,
    t_r: float = 5.0,
    sigma: float = 0.2,
    tau: float = 0.1,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Exponentially Modified Gaussian (EMG) peak function.

    Models chromatographic peak tailing and column overload:
        f(t) = (A / (2 * tau)) * exp(sigma^2 / (2 * tau^2) - (t - t_r) / tau) * erfc(...)
    """
    t_arr = np.asarray(t, dtype=np.float64)
    if tau <= 1e-6:
        return amplitude * np.exp(-0.5 * ((t_arr - t_r) / sigma) ** 2)

    arg_exp = (sigma ** 2 / (2.0 * tau ** 2)) - ((t_arr - t_r) / tau)
    arg_erfc = (sigma / (np.sqrt(2.0) * tau)) - ((t_arr - t_r) / (np.sqrt(2.0) * sigma))
    peak = (amplitude / (2.0 * tau)) * np.exp(np.clip(arg_exp, -100.0, 50.0)) * erfc(arg_erfc)
    return np.maximum(0.0, peak)


def generate_baseline_drift(
    n_points: int = 128,
    drift_amplitude: float = 0.04,
    random_state: Optional[Any] = None,
) -> np.ndarray:
    """Generate stochastic low-frequency baseline drift (quadratic polynomial + random walk)."""
    rng = np.random.RandomState(random_state) if not isinstance(random_state, np.random.RandomState) else random_state
    t = np.linspace(0.0, 1.0, n_points)
    c0 = rng.normal(0.0, drift_amplitude)
    c1 = rng.normal(0.0, drift_amplitude)
    c2 = rng.normal(0.0, drift_amplitude)
    drift = c0 + c1 * t + c2 * (t ** 2)
    return drift


def generate_chromatographic_signal(
    t: np.ndarray,
    analyte_conc: float = 1.0,
    matrix_conc: float = 0.0,
    gain: float = 1.0,
    base_noise: float = 0.006,
    apply_horwitz: bool = True,
    random_state: Optional[Any] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a single realistic chromatographic profile with analyte, matrix, and noise."""
    rng = np.random.RandomState(random_state) if not isinstance(random_state, np.random.RandomState) else random_state

    # Analyte elutes at t_r = 5.0
    p_analyte = emg_peak(t, t_r=5.0, sigma=0.25, tau=0.15, amplitude=analyte_conc)
    # Interferent/matrix elutes at t_r = 3.8
    p_matrix = emg_peak(t, t_r=3.8, sigma=0.35, tau=0.25, amplitude=matrix_conc)
    # Drift
    drift = generate_baseline_drift(len(t), drift_amplitude=0.02, random_state=rng)

    # Noise with Horwitz heteroscedasticity
    noise_sigma = base_noise
    if apply_horwitz and analyte_conc > 0.0:
        noise_sigma = base_noise + 0.02 * (analyte_conc ** 0.8492)

    noise = rng.normal(0.0, noise_sigma, size=len(t))
    signal = gain * (p_analyte + p_matrix) + drift + noise

    meta = {
        "analyte_conc": float(analyte_conc),
        "matrix_conc": float(matrix_conc),
        "noise_sigma": float(noise_sigma),
        "gain": float(gain),
    }
    return signal, meta


def generate_synthetic_batch(
    n_samples: int = 20,
    n_points: int = 128,
    conc_range: Tuple[float, float] = (0.02, 10.0),
    matrix_prob: float = 0.35,
    matrix_max_ratio: float = 0.4,
    gain_std: float = 0.04,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a full batch of synthetic chromatograms with concentrations and gains."""
    rng = np.random.RandomState(random_state)
    t = np.linspace(0.0, 10.0, n_points)
    x_list, y_list, gains_list = [], [], []

    for _ in range(n_samples):
        conc = rng.uniform(conc_range[0], conc_range[1])
        mat_conc = rng.uniform(0.0, conc * matrix_max_ratio) if rng.rand() < matrix_prob else 0.0
        gain = rng.normal(1.0, gain_std) if gain_std > 0 else 1.0
        sig, _ = generate_chromatographic_signal(t, analyte_conc=conc, matrix_conc=mat_conc, gain=gain, random_state=rng)
        x_list.append(sig)
        y_list.append(conc)
        gains_list.append(gain)

    return (
        np.array(x_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(gains_list, dtype=np.float32),
    )
