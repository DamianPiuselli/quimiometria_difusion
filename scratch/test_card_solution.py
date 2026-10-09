"""Test script: Diagnosing and fixing the conditioning collapse in 1D-CNN and Diffusion."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from icdc.data.datasets import load_corn_dataset, split_calibration_transfer
from icdc.metrology.metrics import calc_rmsep, calc_r2
from icdc.models.diffusion import DiffusionSchedule, ScalarDenoisingMLP

corn = load_corn_dataset()
splits = split_calibration_transfer(corn["X_m5"], corn["X_mp5"], corn["y"], n_train_source=50, n_transfer_standards=10, random_state=42)

x_tr = splits["X_source_train"]
y_tr = splits["y_source_train"]
x_te = corn["X_m5"][splits["test_idx"]]
y_te = splits["y_target_test"]

print(f"y_test range: [{y_te.min():.2f}, {y_te.max():.2f}], std: {np.std(y_te):.3f}")

# 1. Standardize X per wavelength channel
scaler = StandardScaler()
sx_tr = scaler.fit_transform(x_tr)
sx_te = scaler.transform(x_te)

# 2. Linear / PLS / Ridge backbone
backbone = Ridge(alpha=10.0).fit(sx_tr, y_tr)
y_pred_tr_bb = backbone.predict(sx_tr)
y_pred_te_bb = backbone.predict(sx_te)
print(f"Backbone (Ridge) R2: {calc_r2(y_te, y_pred_te_bb):.4f}, RMSEP: {calc_rmsep(y_te, y_pred_te_bb):.4f}")

# Normalize y
y_mean, y_std = y_tr.mean(), y_tr.std()
y_tr_norm = (y_tr - y_mean) / y_std
y_pred_tr_norm = (y_pred_tr_bb - y_mean) / y_std
y_pred_te_norm = (y_pred_te_bb - y_mean) / y_std

# 3. CARD Diffusion with Backbone Conditioning
# In the original CARD paper (Han et al., 2022), conditioning on the deterministic prediction y_hat
# provides the necessary anchor to prevent conditioning collapse!
feature_dim = 32 + 1  # 32 projected spectral features + 1 deterministic prediction
context_dim = 16
time_dim = 32

proj = nn.Linear(700, 32)
denoiser = ScalarDenoisingMLP(feature_dim=feature_dim, context_dim=context_dim, time_dim=time_dim, hidden_dim=64)
schedule = DiffusionSchedule(timesteps=50)

opt = torch.optim.AdamW(list(proj.parameters()) + list(denoiser.parameters()), lr=1e-3, weight_decay=1e-4)
crit = nn.MSELoss()

xt_tr = torch.from_numpy(sx_tr).float()
yt_tr = torch.from_numpy(y_tr_norm).float().unsqueeze(1)
ybb_tr = torch.from_numpy(y_pred_tr_norm).float().unsqueeze(1)
c_day_tr = torch.zeros(len(xt_tr), context_dim)

for epoch in range(120):
    t = torch.randint(0, 50, (len(xt_tr),))
    noise = torch.randn_like(yt_tr)
    sqrt_a = schedule.sqrt_alphas_cumprod[t].unsqueeze(1)
    sqrt_1_a = schedule.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
    yt = sqrt_a * yt_tr + sqrt_1_a * noise

    cx = torch.cat([proj(xt_tr), ybb_tr], dim=-1)
    pred_noise = denoiser(yt, t, cx, c_day_tr)

    loss = crit(pred_noise, noise)
    opt.zero_grad()
    loss.backward()
    opt.step()

# Posterior Sampling
xt_te = torch.from_numpy(sx_te).float()
ybb_te = torch.from_numpy(y_pred_te_norm).float().unsqueeze(1)
c_day_te = torch.zeros(len(xt_te), context_dim)
n_test = len(xt_te)

proj.eval()
denoiser.eval()
all_preds = []
with torch.no_grad():
    cx_te = torch.cat([proj(xt_te), ybb_te], dim=-1)
    for _ in range(60):
        y_t = torch.randn(n_test, 1)
        for step in reversed(range(50)):
            t = torch.full((n_test,), step, dtype=torch.long)
            p_noise = denoiser(y_t, t, cx_te, c_day_te)
            beta_t = schedule.betas[step]
            alpha_t = schedule.alphas[step]
            alpha_bar_t = schedule.alphas_cumprod[step]
            mean = (1.0 / torch.sqrt(alpha_t)) * (y_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * p_noise)
            if step > 0:
                y_t = mean + torch.sqrt(schedule.posterior_variance[step]) * torch.randn_like(y_t)
            else:
                y_t = mean
        y_real = (y_t.numpy().ravel() * y_std) + y_mean
        all_preds.append(y_real)

samples = np.column_stack(all_preds)
y_median = np.median(samples, axis=1)
r2_val = calc_r2(y_te, y_median)
rmsep_val = calc_rmsep(y_te, y_median)
print(f"CARD Diffusion (with Ridge anchor): R2 = {r2_val:.4f}, RMSEP = {rmsep_val:.4f}")
print(f"True y_test:    {np.round(y_te[:6], 3)}")
print(f"Predicted y_med: {np.round(y_median[:6], 3)}")
print(f"Pred range: [{y_median.min():.2f}, {y_median.max():.2f}]")
