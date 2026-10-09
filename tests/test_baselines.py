import numpy as np
from icdc.data.synthetic import generate_synthetic_batch
from icdc.models.pls_baseline import PLSBaseline
from icdc.models.cnn_baseline import CNN1DBaseline
from icdc.metrology.metrics import calc_rmsep, calc_r2


def test_pls_baseline():
    x_train, y_train, _ = generate_synthetic_batch(n_samples=40, n_points=64, random_state=1)
    x_test, y_test, _ = generate_synthetic_batch(n_samples=20, n_points=64, random_state=2)

    pls = PLSBaseline(max_components=5, use_snv=False)
    pls.fit(x_train, y_train)

    y_pred, y_low, y_up = pls.predict_intervals(x_test, coverage=0.95)

    assert y_pred.shape == y_test.shape
    assert y_low.shape == y_test.shape
    assert y_up.shape == y_test.shape
    assert np.all(y_up >= y_low)

    r2 = calc_r2(y_test, y_pred)
    assert r2 > 0.5


def test_cnn_baseline():
    x_train, y_train, _ = generate_synthetic_batch(n_samples=30, n_points=64, random_state=1)
    x_test, y_test, _ = generate_synthetic_batch(n_samples=15, n_points=64, random_state=2)

    cnn = CNN1DBaseline(feature_dim=16, epochs=5, batch_size=16, device="cpu")
    cnn.fit(x_train, y_train)

    y_pred, y_low, y_up = cnn.predict_intervals(x_test, coverage=0.95)

    assert y_pred.shape == y_test.shape
    assert y_low.shape == y_test.shape
    assert y_up.shape == y_test.shape
