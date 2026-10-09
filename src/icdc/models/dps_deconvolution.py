"""Diffusion Posterior Sampling (DPS) for Zero-Shot Blind Spectral Deconvolution.

Solves the inverse problem of recovering clean analyte spectra and isolating unmodeled
chemical interferents/adulterants without retraining:
    x_obs = x_clean + x_interferent + noise

Classical methods (PLS) fail silently because the unmodeled interferent projects onto
the regression vector, severely biasing the concentration. DPS uses a generative spectral
prior p(x_clean) and guides reverse diffusion with physical non-negativity constraints
(x_interferent = x_obs - x_clean >= 0), isolating the adulterant signature.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from icdc.models.diffusion import SinusoidalTimeEmbedding, DiffusionSchedule


@dataclass
class DeconvolutionResult:
    """Encapsulates the blind deconvolution decomposition."""

    x_observed: np.ndarray
    x_clean_recovered: np.ndarray
    x_interferent_isolated: np.ndarray
    anomaly_score: np.ndarray
    is_interferent_detected: np.ndarray
    detection_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_anomaly_score": float(np.mean(self.anomaly_score)),
            "detection_rate": float(np.mean(self.is_interferent_detected)),
            "detection_threshold": self.detection_threshold,
        }


class SpectralResBlock1D(nn.Module):
    """1D Residual convolution block with time embedding modulation."""

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.norm1 = nn.GroupNorm(min(4, channels), channels)
        self.time_proj = nn.Linear(time_dim, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.norm2 = nn.GroupNorm(min(4, channels), channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        # Add projected time embedding across channels
        h = h + self.time_proj(t_emb).unsqueeze(-1)
        h = self.act(self.norm2(self.conv2(h)))
        return x + h


class SpectralDenoiser1D(nn.Module):
    """Score matching network for 1D continuous spectral signals."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 64, num_blocks: int = 4, time_dim: int = 32):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3)

        self.blocks = nn.ModuleList([
            SpectralResBlock1D(hidden_dim, time_dim) for _ in range(num_blocks)
        ])

        self.out_conv = nn.Sequential(
            nn.GroupNorm(min(4, hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, in_channels, kernel_size=5, padding=2),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.in_conv(x_t)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out_conv(h)


class SpectralDiffusionDPS:
    """1D Spectral Generative Diffusion with DPS Blind Deconvolution."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        timesteps: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 80,
        batch_size: int = 32,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.timesteps = timesteps
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = SpectralDenoiser1D(
            in_channels=1,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            time_dim=32,
        ).to(self.device)

        self.schedule = DiffusionSchedule(timesteps=timesteps, device=self.device)

        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None

    def fit(self, x_clean: np.ndarray) -> "SpectralDiffusionDPS":
        """Train the spectral score-based generative model on clean calibration spectra."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        x_arr = np.asarray(x_clean, dtype=np.float32)
        if x_arr.ndim == 2:
            # (n_samples, n_points) -> (n_samples, 1, n_points)
            x_arr = x_arr[:, np.newaxis, :]

        # Robust standardization per channel to ensure O(1) variance for diffusion
        self.x_mean = np.mean(x_arr, axis=0, keepdims=True)
        self.x_std = np.std(x_arr, axis=0, keepdims=True)
        self.x_std = np.where(self.x_std < 1e-4, 1.0, self.x_std)

        x_norm = (x_arr - self.x_mean) / self.x_std

        dataset = TensorDataset(torch.from_numpy(x_norm).float())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                t = torch.randint(0, self.timesteps, (batch_x.size(0),), device=self.device)
                noise = torch.randn_like(batch_x)

                sqrt_alpha = self.schedule.sqrt_alphas_cumprod[t].unsqueeze(1).unsqueeze(2)
                sqrt_one_minus_alpha = self.schedule.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1).unsqueeze(2)
                x_t = sqrt_alpha * batch_x + sqrt_one_minus_alpha * noise

                pred_noise = self.model(x_t, t)
                loss = criterion(pred_noise, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    @torch.no_grad()
    def compute_atypicity_score(self, x: np.ndarray, t_step: int = 1) -> np.ndarray:
        """Compute Matrix Atypicity Score via score norm: ||nabla_X log p(X)||.

        Matches the energy-based OOD metric defined in Section 3.3 of README:
            Score = ||eps_theta(x_norm, t)||_2
        Evaluates the magnitude of restoring force back to the clean chemical manifold.
        """
        if self.x_mean is None or self.x_std is None:
            raise RuntimeError("Model has not been fitted.")

        self.model.eval()
        x_raw = np.asarray(x, dtype=np.float32)
        if x_raw.ndim == 2:
            x_arr = x_raw[:, np.newaxis, :]
        else:
            x_arr = x_raw

        n_samples = x_arr.shape[0]
        x_norm = torch.from_numpy((x_arr - self.x_mean) / self.x_std).float().to(self.device)
        t_batch = torch.full((n_samples,), t_step, device=self.device, dtype=torch.long)
        pred_noise = self.model(x_norm, t_batch)
        score_norms = torch.norm(pred_noise.view(n_samples, -1), dim=1).cpu().numpy()
        return score_norms

    def deconvolve(
        self,
        x_observed: np.ndarray,
        t_start: Optional[int] = None,
        guidance_scale: float = 0.6,
        positivity_penalty: float = 3.0,
        detection_threshold: float = 2.1,
    ) -> DeconvolutionResult:
        """Perform Zero-Shot Blind Deconvolution using Diffusion Posterior Sampling (DPS).

        Recovers clean spectral component x_clean from x_obs = x_clean + x_interferent.
        Enforces physical non-negativity of additive interferent: x_interferent >= 0.
        """
        if self.x_mean is None or self.x_std is None:
            raise RuntimeError("Model has not been fitted.")

        self.model.eval()

        x_raw = np.asarray(x_observed, dtype=np.float32)
        squeeze_needed = False
        if x_raw.ndim == 2:
            x_arr = x_raw[:, np.newaxis, :]
            squeeze_needed = True
        else:
            x_arr = x_raw

        n_samples = x_arr.shape[0]
        x_obs_norm = torch.from_numpy((x_arr - self.x_mean) / self.x_std).float().to(self.device)

        if t_start is None:
            t_start = min(self.timesteps - 1, 18)

        # SDEdit initialization: forward-diffuse the observation to t_start
        noise_init = torch.randn_like(x_obs_norm)
        sqrt_alpha_start = self.schedule.sqrt_alphas_cumprod[t_start]
        sqrt_one_start = self.schedule.sqrt_one_minus_alphas_cumprod[t_start]
        x_t = sqrt_alpha_start * x_obs_norm + sqrt_one_start * noise_init

        for step in reversed(range(t_start + 1)):
            x_t = x_t.detach().requires_grad_(True)
            t = torch.full((n_samples,), step, device=self.device, dtype=torch.long)

            pred_noise = self.model(x_t, t)

            beta_t = self.schedule.betas[step]
            alpha_t = self.schedule.alphas[step]
            alpha_bar_t = self.schedule.alphas_cumprod[step]

            # Tweedie's estimator for clean x_0
            x_0_hat = (x_t - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

            # Measurement guidance loss
            # 1. Physical non-negativity: clean spectrum must not exceed observation (interferent >= 0)
            overshoot = torch.relu(x_0_hat - x_obs_norm)
            loss_overshoot = positivity_penalty * torch.mean(overshoot ** 2)

            # 2. Localized interferent sparsity: penalize positive residual
            residual = x_obs_norm - x_0_hat
            loss_sparse = 0.05 * torch.mean(torch.relu(residual))

            total_loss = loss_overshoot + loss_sparse
            guidance_grad = torch.autograd.grad(total_loss, x_t)[0]

            # Normalize gradient per sample to stabilize reverse trajectory
            g_flat = guidance_grad.view(n_samples, -1)
            g_norm = torch.norm(g_flat, dim=1, keepdim=True).unsqueeze(-1)
            g_norm = torch.clamp(g_norm, min=1e-6)
            guidance_step = guidance_grad / g_norm

            with torch.no_grad():
                mean = (1.0 / torch.sqrt(alpha_t)) * (
                    x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
                )
                # Apply scaled DPS guidance
                mean = mean - guidance_scale * (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * guidance_step

                if step > 0:
                    z = torch.randn_like(x_t)
                    var = self.schedule.posterior_variance[step]
                    x_t = mean + torch.sqrt(var) * z
                else:
                    x_t = x_0_hat.detach()

        # Final clean reconstruction de-standardized
        x_clean_rec = (x_t.detach().cpu().numpy() * self.x_std + self.x_mean)
        if np.all(x_raw >= 0):
            x_clean_rec = np.maximum(0.0, x_clean_rec)

        # Isolated interferent: x_interferent = max(0, x_obs - x_clean)
        x_obs_np = x_raw[:, np.newaxis, :] if squeeze_needed else x_raw
        x_interferent = np.maximum(0.0, x_obs_np - x_clean_rec)

        # Compute Matrix Atypicity Score via score norm: ||nabla_X log p(X)||
        anomaly_scores = self.compute_atypicity_score(x_raw)
        is_detected = anomaly_scores >= detection_threshold

        if squeeze_needed:
            x_clean_rec = x_clean_rec.squeeze(1)
            x_interferent = x_interferent.squeeze(1)

        return DeconvolutionResult(
            x_observed=x_raw,
            x_clean_recovered=x_clean_rec,
            x_interferent_isolated=x_interferent,
            anomaly_score=anomaly_scores,
            is_interferent_detected=is_detected,
            detection_threshold=detection_threshold,
        )
