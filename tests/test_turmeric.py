"""Tests for Turmeric FT-Raman dataset loader."""

from pathlib import Path
import numpy as np
from icdc.datasets import load_turmeric_dataset


def test_turmeric_dataset_loading():
    data = load_turmeric_dataset()

    assert "X_laced" in data
    assert "y_conc_pct" in data
    assert "y_conc_mg_kg" in data
    assert "sample_ids" in data
    assert "wavenumbers" in data
    assert "X_authentic" in data
    assert "X_pure_adulterant" in data

    assert data["X_laced"].shape == (70, 805)
    assert data["y_conc_pct"].shape == (70,)
    assert data["X_authentic"].shape == (15, 805)
    assert data["X_pure_adulterant"].shape == (3, 805)
    assert data["wavenumbers"].shape == (805,)

    # Verify concentration levels
    expected_concs = np.array([0.0, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0])
    np.testing.assert_allclose(np.unique(data["y_conc_pct"]), expected_concs, atol=1e-5)
