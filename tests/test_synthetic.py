import numpy as np
from icdc.data.synthetic import (
    emg_peak,
    generate_chromatographic_signal,
    generate_synthetic_batch,
)


def test_emg_peak():
    t = np.linspace(0.0, 10.0, 500)
    peak = emg_peak(t, t_r=5.0, sigma=0.2, tau=0.1, amplitude=2.0)

    assert peak.shape == t.shape
    assert np.all(peak >= 0.0)
    assert np.max(peak) > 0.5
    # Peak maximum should be slightly shifted right from t_r due to exponential tailing
    max_idx = np.argmax(peak)
    assert t[max_idx] >= 5.0


def test_chromatographic_signal():
    t = np.linspace(0.0, 10.0, 256)
    signal, meta = generate_chromatographic_signal(t, analyte_conc=1.5, matrix_conc=0.5, gain=1.0)

    assert signal.shape == t.shape
    assert meta["analyte_conc"] == 1.5
    assert meta["matrix_conc"] == 0.5
    assert meta["noise_sigma"] > 0.0


def test_synthetic_batch():
    x, y, gains = generate_synthetic_batch(n_samples=20, n_points=128, random_state=42)

    assert x.shape == (20, 128)
    assert y.shape == (20,)
    assert gains.shape == (20,)
    assert np.all(y > 0.0)
