"""PLS Baseline Regressor for Chemometrics with Cross-Validation and SNV preprocessing."""

from typing import Optional, Tuple
import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold


def apply_snv(x: np.ndarray) -> np.ndarray:
    """Standard Normal Variate (SNV) transformation across spectral channels."""
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (x - mean) / std


class PLSBaseline:
    """Partial Least Squares (PLS) Regressor with automated LV selection via CV.

    Supports native chemometric preprocessing: Savitzky-Golay filtering/derivatives and SNV.
    Provides point predictions and classical parametric confidence intervals
    based on residual standard error (for GUM comparison).
    """

    def __init__(
        self,
        max_components: int = 15,
        use_snv: bool = True,
        savgol_window: Optional[int] = None,
        savgol_poly: int = 2,
        savgol_deriv: int = 0,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        self.max_components = max_components
        self.use_snv = use_snv
        self.savgol_window = savgol_window
        self.savgol_poly = savgol_poly
        self.savgol_deriv = savgol_deriv
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_n_components: Optional[int] = None
        self.model: Optional[PLSRegression] = None
        self.residual_variance: Optional[float] = None

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        """Apply configured chemometric preprocessing (Savitzky-Golay and/or SNV)."""
        x_proc = np.asarray(x, dtype=np.float64).copy()
        if self.savgol_window is not None and self.savgol_window > 2:
            # Window length must be odd and <= number of channels
            w = self.savgol_window if self.savgol_window % 2 == 1 else self.savgol_window + 1
            w = min(w, x_proc.shape[-1] if x_proc.shape[-1] % 2 == 1 else x_proc.shape[-1] - 1)
            if w > self.savgol_poly:
                x_proc = savgol_filter(x_proc, window_length=w, polyorder=self.savgol_poly, deriv=self.savgol_deriv, axis=-1)

        if self.use_snv:
            x_proc = apply_snv(x_proc)

        return x_proc

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PLSBaseline":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        x_proc = self.preprocess(x)

        n_samples = x.shape[0]
        max_lv = min(self.max_components, n_samples - 2, x.shape[1])
        max_lv = max(1, max_lv)

        # Cross-validation for optimal Latent Variables (LV)
        kf = KFold(n_splits=min(self.cv_folds, n_samples), shuffle=True, random_state=self.random_state)
        cv_errors = []

        candidate_lvs = list(range(1, max_lv + 1))
        for n_comp in candidate_lvs:
            fold_errors = []
            for train_idx, val_idx in kf.split(x_proc):
                pls = PLSRegression(n_components=n_comp)
                pls.fit(x_proc[train_idx], y[train_idx])
                y_val_pred = pls.predict(x_proc[val_idx])
                fold_errors.append(np.mean((y[val_idx] - y_val_pred) ** 2))
            cv_errors.append(np.mean(fold_errors))

        self.best_n_components = candidate_lvs[int(np.argmin(cv_errors))]

        # Fit final model on full training set
        self.model = PLSRegression(n_components=self.best_n_components)
        self.model.fit(x_proc, y)

        # Compute residual standard deviation
        y_train_pred = self.model.predict(x_proc)
        residuals = y.ravel() - y_train_pred.ravel()
        dof = max(1, n_samples - self.best_n_components - 1)
        self.residual_variance = float(np.sum(residuals ** 2) / dof)

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict scalar target y."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        x = np.asarray(x, dtype=np.float64)
        x_proc = self.preprocess(x)
        return self.model.predict(x_proc).ravel()

    def predict_intervals(
        self, x: np.ndarray, coverage: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict point estimate along with parametric confidence intervals.

        Assumes homoscedastic Gaussian residual error (the standard linear assumption).
        Returns: (y_pred, y_lower, y_upper)
        """
        from scipy import stats

        y_pred = self.predict(x)
        se = np.sqrt(self.residual_variance) if self.residual_variance is not None else 1.0
        z_crit = stats.norm.ppf(0.5 + coverage / 2.0)
        margin = z_crit * se
        y_lower = y_pred - margin
        y_upper = y_pred + margin
        return y_pred, y_lower, y_upper
