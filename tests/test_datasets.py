from icdc.data.datasets import load_corn_dataset, split_calibration_transfer


def test_corn_dataset_loading_and_split():
    data = load_corn_dataset(data_dir="data/corn")

    assert "X_m5" in data
    assert "X_mp5" in data
    assert "X_mp6" in data
    assert "y" in data

    assert data["X_m5"].shape == (80, 700)
    assert data["X_mp5"].shape == (80, 700)
    assert data["y"].shape == (80,)

    # Test calibration transfer split
    split = split_calibration_transfer(
        data["X_m5"],
        data["X_mp5"],
        data["y"],
        n_train_source=50,
        n_transfer_standards=10,
    )

    assert split["X_source_train"].shape == (50, 700)
    assert split["X_target_transfer"].shape == (10, 700)
    assert split["X_target_test"].shape == (20, 700)
    assert split["y_target_test"].shape == (20,)
