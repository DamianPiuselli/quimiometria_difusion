"""Deterministic 1D-CNN Baseline Regressor for Chemometrics."""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class Conv1DBackbone(nn.Module):
    """1D-CNN Backbone for spectral feature extraction."""

    def __init__(self, in_channels: int = 1, feature_dim: int = 64):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.conv_net(x)


class CNN1DRegressorNet(nn.Module):
    """Full 1D-CNN Regressor network."""

    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.backbone = Conv1DBackbone(in_channels=1, feature_dim=feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


class CNN1DBaseline:
    """Scikit-learn style wrapper for the 1D-CNN Regressor."""

    def __init__(
        self,
        feature_dim: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 32,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.feature_dim = feature_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.net: Optional[CNN1DRegressorNet] = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0
        self.residual_variance: Optional[float] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CNN1DBaseline":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()

        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y)) if np.std(y) > 0 else 1.0
        y_norm = (y - self.y_mean) / self.y_std

        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y_norm).unsqueeze(1))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net = CNN1DRegressorNet(feature_dim=self.feature_dim).to(self.device)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        self.net.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                pred = self.net(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

        # Compute residual variance
        y_pred = self.predict(x)
        res = y - y_pred
        self.residual_variance = float(np.mean(res ** 2))

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Model has not been fitted.")

        self.net.eval()
        x_t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            preds_norm = self.net(x_t).cpu().numpy().ravel()
        return (preds_norm * self.y_std) + self.y_mean

    def predict_intervals(
        self, x: np.ndarray, coverage: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with homoscedastic residual error intervals (standard deep learning baseline)."""
        from scipy import stats

        y_pred = self.predict(x)
        se = np.sqrt(self.residual_variance) if self.residual_variance is not None else 1.0
        z_crit = stats.norm.ppf(0.5 + coverage / 2.0)
        margin = z_crit * se
        return y_pred, y_pred - margin, y_pred + margin
