import numpy as np
from icdc.data.synthetic import generate_synthetic_batch
from icdc.models.diffusion import DiffusionRegressor
from icdc.metrology.metrics import calc_rmsep, calc_picp


def test_diffusion_regressor_training_and_sampling():
    x_train, y_train, _ = generate_synthetic_batch(n_samples=30, n_points=64, random_state=1)
    x_test, y_test, _ = generate_synthetic_batch(n_samples=10, n_points=64, random_state=2)

    model = DiffusionRegressor(
        spectral_dim=16,
        context_dim=8,
        timesteps=20,
        epochs=10,
        batch_size=16,
        device="cpu",
    )
    model.fit(x_train, y_train)

    # Sample posterior
    samples = model.sample_posterior(x_test, n_samples=15)
    assert samples.shape == (10, 15)
    assert np.all(samples >= 0.0)

    # Predict intervals
    y_pred, y_low, y_up = model.predict_intervals(x_test, coverage=0.90, n_samples=15)
    assert y_pred.shape == y_test.shape
    assert y_low.shape == y_test.shape
    assert y_up.shape == y_test.shape
    assert np.all(y_up >= y_low)

    # Check non-negativity
    assert np.all(y_low >= 0.0)
