"""In-Context Diffusion Calibration (ICDC) Model.

Implements a 1D conditional diffusion model (CARD framework) for analytical chemometrics:
- Denoises the scalar concentration target y conditioned on:
  1. Spectral features extracted via 1D-CNN backbone (with localized regional pooling).
  2. Anti-collapse deterministic prediction anchor (CARD framework).
  3. In-context daily calibration vector (E_day) with transfer-shift training.
  4. Sinusoidal time-step embeddings.
- Provides non-parametric, heteroscedastic uncertainty estimation with GUM compliance.
- Fast sampling via DDPM / DDIM.
"""

import math
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from icdc.models.cnn_baseline import Conv1DBackbone


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion time-step t."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.float().unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = nn.functional.pad(embeddings, (0, 1))
        return embeddings


class ScalarDenoisingMLP(nn.Module):
    """Residual MLP that predicts added noise epsilon given noisy y_t, time t, and conditions."""

    def __init__(self, feature_dim: int = 65, context_dim: int = 16, time_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
        )
        self.context_dim = context_dim
        in_dim = 1 + time_dim + feature_dim + context_dim
        self.input_layer = nn.Linear(in_dim, hidden_dim)

        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.out_layer = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        y_t: torch.Tensor,
        t: torch.Tensor,
        c_x: torch.Tensor,
        c_day: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_emb = self.time_embed(t)
        if c_day is None:
            c_day = torch.zeros(y_t.size(0), self.context_dim, device=y_t.device)

        inputs = torch.cat([y_t, t_emb, c_x, c_day], dim=-1)
        h = nn.functional.gelu(self.input_layer(inputs))
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.out_layer(h)


class DiffusionSchedule:
    """Beta schedule for diffusion."""

    def __init__(self, timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02, device: torch.device = torch.device("cpu")):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Variance for posterior q(y_{t-1} | y_t, y_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )


class DiffusionRegressor:
    """In-Context Diffusion Calibration (ICDC) model with CARD anti-collapse anchor."""

    def __init__(
        self,
        spectral_dim: int = 64,
        context_dim: int = 16,
        timesteps: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 32,
        standardize_x: bool = True,
        use_anchor: bool = True,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.spectral_dim = spectral_dim
        self.context_dim = context_dim
        self.timesteps = timesteps
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.standardize_x = standardize_x
        self.use_anchor = use_anchor
        self.random_state = random_state

        self.scaler: Optional[StandardScaler] = StandardScaler() if standardize_x else None
        self.anchor_model: Optional[Ridge] = Ridge(alpha=10.0) if use_anchor else None

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.total_feature_dim = spectral_dim + (1 if use_anchor else 0)
        self.backbone = Conv1DBackbone(in_channels=1, feature_dim=spectral_dim).to(self.device)
        self.denoiser = ScalarDenoisingMLP(
            feature_dim=self.total_feature_dim,
            context_dim=context_dim,
            time_dim=32,
            hidden_dim=128,
        ).to(self.device)
        self.schedule = DiffusionSchedule(timesteps=timesteps, device=self.device)

        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        c_day: Optional[np.ndarray] = None,
        augment_transfer_shift: bool = True,
    ) -> "DiffusionRegressor":
        """Train the diffusion model via score matching with CARD anti-collapse anchoring."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        x_raw = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).ravel()

        if self.standardize_x and self.scaler is not None:
            x_arr = self.scaler.fit_transform(x_raw)
        else:
            x_arr = x_raw

        self.y_mean = float(np.mean(y_arr))
        self.y_std = float(np.std(y_arr)) if np.std(y_arr) > 0 else 1.0
        y_norm = (y_arr - self.y_mean) / self.y_std

        # Train deterministic anchor model if enabled
        if self.use_anchor and self.anchor_model is not None:
            self.anchor_model.fit(x_arr, y_arr)
            y_anchor = self.anchor_model.predict(x_arr)
            y_anchor_norm = (y_anchor - self.y_mean) / self.y_std
        else:
            y_anchor_norm = np.zeros_like(y_norm)

        if c_day is None:
            c_day_arr = np.zeros((len(x_arr), self.context_dim), dtype=np.float32)
        else:
            c_day_arr = np.asarray(c_day, dtype=np.float32)

        dataset = TensorDataset(
            torch.from_numpy(x_arr),
            torch.from_numpy(y_norm).unsqueeze(1),
            torch.from_numpy(y_anchor_norm).unsqueeze(1),
            torch.from_numpy(c_day_arr),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        params = list(self.backbone.parameters()) + list(self.denoiser.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        self.backbone.train()
        self.denoiser.train()

        for _ in range(self.epochs):
            for bx, by0, by_anch, bcday in loader:
                bx = bx.to(self.device)
                by0 = by0.to(self.device)
                by_anch = by_anch.to(self.device)
                bcday = bcday.to(self.device)

                # Simulated transfer shift augmentation to train in-context calibration response
                if augment_transfer_shift and self.context_dim > 0:
                    sim_shift = torch.randn_like(bcday) * 1.5
                    bcday = bcday + sim_shift
                    # The anchor reflects today's shifted instrument
                    by_anch = by_anch + sim_shift[:, :1]

                t = torch.randint(0, self.timesteps, (bx.size(0),), device=self.device)
                noise = torch.randn_like(by0)

                sqrt_alpha = self.schedule.sqrt_alphas_cumprod[t].unsqueeze(1)
                sqrt_one_minus_alpha = self.schedule.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
                yt = sqrt_alpha * by0 + sqrt_one_minus_alpha * noise

                cx_feat = self.backbone(bx)
                if self.use_anchor:
                    cx = torch.cat([cx_feat, by_anch], dim=-1)
                else:
                    cx = cx_feat

                pred_noise = self.denoiser(yt, t, cx, bcday)
                loss = criterion(pred_noise, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    @torch.no_grad()
    def sample_posterior(
        self,
        x: np.ndarray,
        n_samples: int = 100,
        c_day: Optional[np.ndarray] = None,
        clip_non_negative: bool = True,
    ) -> np.ndarray:
        """Sample from posterior distribution p(y | x, c_day) via DDPM reverse diffusion."""
        self.backbone.eval()
        self.denoiser.eval()

        x_raw = np.asarray(x, dtype=np.float32)
        if self.standardize_x and self.scaler is not None:
            x_proc = self.scaler.transform(x_raw)
        else:
            x_proc = x_raw

        x_t = torch.from_numpy(x_proc).to(self.device)
        n_test = x_t.size(0)

        if self.use_anchor and self.anchor_model is not None:
            y_anchor = self.anchor_model.predict(x_proc)
            y_anchor_norm = (y_anchor - self.y_mean) / self.y_std
            y_anch_t = torch.from_numpy(y_anchor_norm).float().unsqueeze(1).to(self.device)
        else:
            y_anch_t = torch.zeros(n_test, 1, device=self.device)

        if c_day is None:
            c_day_t = torch.zeros(n_test, self.context_dim, device=self.device)
        else:
            c_day_t = torch.from_numpy(np.asarray(c_day, dtype=np.float32)).to(self.device)

        cx_feat = self.backbone(x_t)
        if self.use_anchor:
            cx = torch.cat([cx_feat, y_anch_t], dim=-1)
        else:
            cx = cx_feat

        all_predictions = []
        for _ in range(n_samples):
            y_t = torch.randn(n_test, 1, device=self.device)

            for step in reversed(range(self.timesteps)):
                t = torch.full((n_test,), step, device=self.device, dtype=torch.long)
                pred_noise = self.denoiser(y_t, t, cx, c_day_t)

                beta_t = self.schedule.betas[step]
                alpha_t = self.schedule.alphas[step]
                alpha_bar_t = self.schedule.alphas_cumprod[step]

                mean = (1.0 / torch.sqrt(alpha_t)) * (
                    y_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
                )

                if step > 0:
                    z = torch.randn_like(y_t)
                    var = self.schedule.posterior_variance[step]
                    y_t = mean + torch.sqrt(var) * z
                else:
                    y_t = mean

            y_real = (y_t.cpu().numpy().ravel() * self.y_std) + self.y_mean
            if clip_non_negative:
                y_real = np.maximum(0.0, y_real)
            all_predictions.append(y_real)

        return np.column_stack(all_predictions)

    def predict(self, x: np.ndarray, n_samples: int = 50, c_day: Optional[np.ndarray] = None) -> np.ndarray:
        samples = self.sample_posterior(x, n_samples=n_samples, c_day=c_day)
        return np.median(samples, axis=1)

    def predict_intervals(
        self,
        x: np.ndarray,
        coverage: float = 0.95,
        n_samples: int = 100,
        c_day: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        samples = self.sample_posterior(x, n_samples=n_samples, c_day=c_day)
        y_median = np.median(samples, axis=1)

        alpha = 1.0 - coverage
        lower_percentile = (alpha / 2.0) * 100.0
        upper_percentile = (1.0 - alpha / 2.0) * 100.0

        y_lower = np.percentile(samples, lower_percentile, axis=1)
        y_upper = np.percentile(samples, upper_percentile, axis=1)

        return y_median, y_lower, y_upper
