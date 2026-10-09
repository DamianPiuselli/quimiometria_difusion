"""Standard Chemometric Benchmark Datasets."""

from pathlib import Path
from typing import Dict, Any, Optional
import urllib.request
import numpy as np
import scipy.io


CORN_URL = "http://www.eigenvector.com/data/Corn/corn.mat"


def load_corn_dataset(data_dir: str = "data/corn") -> Dict[str, Any]:
    """Load Cargill Corn NIR multi-instrument dataset (M5, MP5, MP6).

    Property predicted: Moisture (column 0 of propvals, range 9.38% - 10.83%).
    """
    path = Path(data_dir) / "corn.mat"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Corn NIR dataset to {path}...")
        urllib.request.urlretrieve(CORN_URL, path)

    mat = scipy.io.loadmat(path)

    # Extract spectra matrices (80 samples x 700 wavelengths: 1100-2498 nm)
    def _extract(key):
        val = mat[key]
        if hasattr(val, "dtype") and val.dtype.names and "data" in val.dtype.names:
            return val["data"][0, 0].astype(np.float32)
        return val.astype(np.float32)

    x_m5 = _extract("m5spec")
    x_mp5 = _extract("mp5spec")
    x_mp6 = _extract("mp6spec")
    prop = _extract("propvals")

    # Column 0 is moisture
    y = prop[:, 0].astype(np.float32)
    wavelengths = np.linspace(1100.0, 2498.0, x_m5.shape[1], dtype=np.float32)

    return {
        "X_m5": x_m5,
        "X_mp5": x_mp5,
        "X_mp6": x_mp6,
        "y": y,
        "wavelengths": wavelengths,
        "prop_all": prop,
    }


def split_calibration_transfer(
    x_source: np.ndarray,
    x_target: np.ndarray,
    y: np.ndarray,
    n_train_source: int = 50,
    n_transfer_standards: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Split dataset for cross-instrument calibration transfer evaluation."""
    n_total = len(y)
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n_total)

    train_idx = perm[:n_train_source]
    transfer_idx = perm[n_train_source : n_train_source + n_transfer_standards]
    test_idx = perm[n_train_source + n_transfer_standards :]

    return {
        "X_source_train": x_source[train_idx],
        "y_source_train": y[train_idx],
        "X_target_transfer": x_target[transfer_idx],
        "y_target_transfer": y[transfer_idx],
        "X_target_test": x_target[test_idx],
        "y_target_test": y[test_idx],
        "train_idx": train_idx,
        "transfer_idx": transfer_idx,
        "test_idx": test_idx,
    }
