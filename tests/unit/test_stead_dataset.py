import h5py
import numpy as np
import pandas as pd
import pytest

from seismopulse.data.stead_dataset import STEADDataset


@pytest.fixture
def sample_dataset(tmp_path):
    csv_path = tmp_path / "metadata.csv"
    hdf5_path = tmp_path / "waveforms.hdf5"

    metadata = pd.DataFrame({
        "trace_name": ["trace_001", "trace_002"],
        "trace_category": ["earthquake_local", "noise"],
        "trace_sampling_rate": [100, 100],
        "trace_p_arrival_sample": [120, None],
        "trace_s_arrival_sample": [250, None],
    })

    metadata.to_csv(csv_path, index=False)

    with h5py.File(hdf5_path, "w") as f:
        group = f.create_group("data")
        group.create_dataset(
            "trace_001",
            data=np.random.default_rng(42).normal(
                size=(6000, 3)
            ).astype(np.float32),
        )
        group.create_dataset(
            "trace_002",
            data=np.random.default_rng(43).normal(
                size=(6000, 3)
            ).astype(np.float32),
        )

    return csv_path, hdf5_path


def test_dataset_length(sample_dataset):
    dataset = STEADDataset(*sample_dataset)
    assert len(dataset) == 2


def test_get_record(sample_dataset):
    dataset = STEADDataset(*sample_dataset)
    record = dataset.get_record("trace_001")

    assert record["trace_name"] == "trace_001"
    assert record["category"] == "earthquake_local"
    assert record["sampling_rate"] == 100
    assert record["waveform"].shape == (3, 6000)


def test_unknown_trace(sample_dataset):
    dataset = STEADDataset(*sample_dataset)

    with pytest.raises(KeyError):
        dataset.get_record("unknown_trace")


def test_duplicate_trace_names_rejected(sample_dataset):
    csv_path, hdf5_path = sample_dataset

    metadata = pd.read_csv(csv_path)
    metadata = pd.concat(
        [metadata, metadata.iloc[[0]]],
        ignore_index=True,
    )
    metadata.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        STEADDataset(csv_path, hdf5_path)