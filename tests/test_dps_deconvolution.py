import numpy as np
import torch
from icdc.data.synthetic import generate_synthetic_batch
from icdc.models.dps_deconvolution import SpectralDiffusionDPS, DeconvolutionResult


def test_dps_deconvolution_fast():
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate small clean synthetic spectra
    n_points = 32
    x_clean, _, _ = generate_synthetic_batch(n_samples=16, n_points=n_points, random_state=42)

    # Train minimal spectral diffusion model
    dps_model = SpectralDiffusionDPS(
        hidden_dim=16,
        num_blocks=2,
        timesteps=10,
        epochs=3,
        batch_size=8,
        device="cpu",
        random_state=42,
    )
    dps_model.fit(x_clean)

    # Create contaminated spectra by adding an artificial interferent peak
    x_test_clean, _, _ = generate_synthetic_batch(n_samples=4, n_points=n_points, random_state=99)
    # Add a localized interferent peak in the middle channels
    interferent = np.zeros_like(x_test_clean)
    interferent[:, 12:18] = 0.5
    x_contaminated = x_test_clean + interferent

    result = dps_model.deconvolve(
        x_contaminated,
        guidance_scale=1.0,
        positivity_penalty=2.0,
        detection_threshold=0.05,
    )

    assert isinstance(result, DeconvolutionResult)
    assert result.x_clean_recovered.shape == x_contaminated.shape
    assert result.x_interferent_isolated.shape == x_contaminated.shape
    assert len(result.anomaly_score) == 4
    assert np.all(result.anomaly_score >= 0.0)
    assert np.all(result.x_interferent_isolated >= 0.0)
