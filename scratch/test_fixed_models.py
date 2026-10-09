"""Diagnosis and verification of the fix for 1D-CNN and ICDC Diffusion on Corn NIR.

Fixes applied:
1. Wavelength-wise standardization (StandardScaler): brings the small +/- 0.01 variance into standard scale.
2. Architecture fix: replaces global-averaging Conv1D (which washes out narrow NIR bands) with a localized
   Spectral Dense/Conv architecture or PLS/Ridge feature projection.
3. Anti-collapse anchor in CARD Diffusion: conditions the denoiser on the deterministic prediction y_hat.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from icdc.data.datasets import load_corn_dataset, split_calibration_transfer
from icdc.models.pls_baseline import PLSBaseline
from icdc.metrology.metrics import calc_rmsep, calc_r2, calc_bias, calc_picp
from icdc.models.diffusion import DiffusionSchedule, ScalarDenoisingMLP

# 1. Load Corn data
corn = load_corn_dataset()
splits = split_calibration_transfer(
    corn["X_m5"], corn["X_mp5"], corn["y"], n_train_source=50, n_transfer_standards=10, random_state=42
)

x_tr = splits["X_source_train"]
y_tr = splits["y_source_train"]
x_te_m5 = corn["X_m5"][splits["test_idx"]]
x_te_mp5 = splits["X_target_test"]
y_te = splits["y_target_test"]
x_transfer = splits["X_target_transfer"]
y_transfer = splits["y_target_transfer"]

n_test = len(y_te)
print(f"y_test: true mean={y_te.mean():.3f}, std={y_te.std():.3f}, range=[{y_te.min():.2f}, {y_te.max():.2f}]")

# Standardize X per wavelength
scaler = StandardScaler()
sx_tr = scaler.fit_transform(x_tr)
sx_te_m5 = scaler.transform(x_te_m5)
sx_te_mp5 = scaler.transform(x_te_mp5)
sx_transfer = scaler.transform(x_transfer)

# Target normalization
y_mean, y_std = y_tr.mean(), y_tr.std()
y_tr_norm = (y_tr - y_mean) / y_std

# A. PLS Model
pls = PLSBaseline(max_components=10, use_snv=False).fit(x_tr, y_tr)
y_pred_m5_pls = pls.predict(x_te_m5)
y_pred_mp5_pls = pls.predict(x_te_mp5)

print("\n--- 1. PLS Results ---")
print(f"PLS (M5 -> M5):   R2 = {calc_r2(y_te, y_pred_m5_pls):.4f}, RMSEP = {calc_rmsep(y_te, y_pred_m5_pls):.4f}, Range = [{y_pred_m5_pls.min():.2f}, {y_pred_m5_pls.max():.2f}]")
print(f"PLS (M5 -> MP5):  R2 = {calc_r2(y_te, y_pred_mp5_pls):.4f}, RMSEP = {calc_rmsep(y_te, y_pred_mp5_pls):.4f}, Bias = {calc_bias(y_te, y_pred_mp5_pls):+.4f}")

# B. Improved Neural Model (Spectral MLP with Residual connections)
print("\n--- 2. Spectral MLP (on standardized X) ---")
class SpectralNet(nn.Module):
    def __init__(self, in_features=700):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)

net = SpectralNet()
opt_net = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
crit = nn.MSELoss()

xt_tr = torch.from_numpy(sx_tr).float()
yt_tr = torch.from_numpy(y_tr_norm).float().unsqueeze(1)

for _ in range(200):
    net.train()
    opt_net.zero_grad()
    loss = crit(net(xt_tr), yt_tr)
    loss.backward()
    opt_net.step()

net.eval()
with torch.no_grad():
    pred_m5_net = (net(torch.from_numpy(sx_te_m5).float()).numpy().ravel() * y_std) + y_mean
    pred_mp5_net = (net(torch.from_numpy(sx_te_mp5).float()).numpy().ravel() * y_std) + y_mean

print(f"SpectralNet (M5 -> M5):  R2 = {calc_r2(y_te, pred_m5_net):.4f}, RMSEP = {calc_rmsep(y_te, pred_m5_net):.4f}, Range = [{pred_m5_net.min():.2f}, {pred_m5_net.max():.2f}]")
print(f"SpectralNet (M5 -> MP5): R2 = {calc_r2(y_te, pred_mp5_net):.4f}, RMSEP = {calc_rmsep(y_te, pred_mp5_net):.4f}, Bias = {calc_bias(y_te, pred_mp5_net):+.4f}")

# C. Fixed CARD Diffusion with Backbone Conditioning and In-Context Transfer
print("\n--- 3. Fixed CARD Diffusion (with PLS/Spectral anchor + In-Context context) ---")
# Use the deterministic prediction as anchor to eliminate conditioning collapse
y_det_tr = (pls.predict(x_tr) - y_mean) / y_std
y_det_te_m5 = (pls.predict(x_te_m5) - y_mean) / y_std
y_det_te_mp5 = (pls.predict(x_te_mp5) - y_mean) / y_std

# In-context daily context: mean difference of transfer standards
c_transfer_bias = float(np.mean(pls.predict(x_transfer) - y_transfer)) / y_std
c_ctx_m5 = torch.zeros(len(x_te_m5), 1)
c_ctx_mp5 = torch.full((len(x_te_mp5), 1), c_transfer_bias)

# Denoising model taking [y_t, t, c_x (projected spectrum + y_det), c_day]
proj_dim = 16
proj = nn.Linear(700, proj_dim)
feature_dim = proj_dim + 1 # spectral proj + y_det
context_dim = 1

denoiser = ScalarDenoisingMLP(feature_dim=feature_dim, context_dim=context_dim, time_dim=32, hidden_dim=64)
schedule = DiffusionSchedule(timesteps=50)

opt_diff = torch.optim.AdamW(list(proj.parameters()) + list(denoiser.parameters()), lr=1e-3, weight_decay=1e-4)

ydet_t_tr = torch.from_numpy(y_det_tr).float().unsqueeze(1)
c_day_tr = torch.zeros(len(xt_tr), context_dim)

for epoch in range(150):
    t = torch.randint(0, 50, (len(xt_tr),))
    noise = torch.randn_like(yt_tr)
    sqrt_a = schedule.sqrt_alphas_cumprod[t].unsqueeze(1)
    sqrt_1_a = schedule.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
    yt = sqrt_a * yt_tr + sqrt_1_a * noise

    cx = torch.cat([proj(xt_tr), ydet_t_tr], dim=-1)
    pred_noise = denoiser(yt, t, cx, c_day_tr)

    loss = crit(pred_noise, noise)
    opt_diff.zero_grad()
    loss.backward()
    opt_diff.step()

# Sample for M5
proj.eval()
denoiser.eval()
with torch.no_grad():
    cx_m5 = torch.cat([proj(torch.from_numpy(sx_te_m5).float()), torch.from_numpy(y_det_te_m5).float().unsqueeze(1)], dim=-1)
    all_preds_m5 = []
    for _ in range(50):
        y_t = torch.randn(len(x_te_m5), 1)
        for step in reversed(range(50)):
            t = torch.full((len(x_te_m5),), step, dtype=torch.long)
            p_noise = denoiser(y_t, t, cx_m5, c_ctx_m5)
            b = schedule.betas[step]
            a = schedule.alphas[step]
            ab = schedule.alphas_cumprod[step]
            m = (1.0 / torch.sqrt(a)) * (y_t - (b / torch.sqrt(1.0 - ab)) * p_noise)
            if step > 0:
                y_t = m + torch.sqrt(schedule.posterior_variance[step]) * torch.randn_like(y_t)
            else:
                y_t = m
        all_preds_m5.append((y_t.numpy().ravel() * y_std) + y_mean)
    diff_m5_samples = np.column_stack(all_preds_m5)
    diff_m5_med = np.median(diff_m5_samples, axis=1)

print(f"Diffusion (M5 -> M5):  R2 = {calc_r2(y_te, diff_m5_med):.4f}, RMSEP = {calc_rmsep(y_te, diff_m5_med):.4f}, Range = [{diff_m5_med.min():.2f}, {diff_m5_med.max():.2f}]")

# Sample for MP5 (transfer)
with torch.no_grad():
    cx_mp5 = torch.cat([proj(torch.from_numpy(sx_te_mp5).float()), torch.from_numpy(y_det_te_mp5).float().unsqueeze(1)], dim=-1)
    all_preds_mp5 = []
    for _ in range(50):
        y_t = torch.randn(len(x_te_mp5), 1)
        for step in reversed(range(50)):
            t = torch.full((len(x_te_mp5),), step, dtype=torch.long)
            p_noise = denoiser(y_t, t, cx_mp5, c_ctx_mp5)
            b = schedule.betas[step]
            a = schedule.alphas[step]
            ab = schedule.alphas_cumprod[step]
            m = (1.0 / torch.sqrt(a)) * (y_t - (b / torch.sqrt(1.0 - ab)) * p_noise)
            if step > 0:
                y_t = m + torch.sqrt(schedule.posterior_variance[step]) * torch.randn_like(y_t)
            else:
                y_t = m
        all_preds_mp5.append((y_t.numpy().ravel() * y_std) + y_mean)
    diff_mp5_samples = np.column_stack(all_preds_mp5)
    diff_mp5_med = np.median(diff_mp5_samples, axis=1)

print(f"Diffusion (M5 -> MP5): R2 = {calc_r2(y_te, diff_mp5_med):.4f}, RMSEP = {calc_rmsep(y_te, diff_mp5_med):.4f}, Bias = {calc_bias(y_te, diff_mp5_med):+.4f}, Range = [{diff_mp5_med.min():.2f}, {diff_mp5_med.max():.2f}]")
