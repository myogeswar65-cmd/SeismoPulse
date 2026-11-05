import h5py
import numpy as np
import pytest
import torch

from seismopulse.data.curated_stead import CuratedSTEADDataset


@pytest.fixture
def sample_hdf5(tmp_path):
    path = tmp_path / "sample.hdf5"

    with h5py.File(path, "w") as hdf:
        hdf.create_dataset(
            "traces",
            data=np.random.default_rng(42).normal(
                size=(3, 6000, 3)
            ).astype(np.float32),
        )

        hdf.create_dataset(
            "p_arrival",
            data=[900, 1000, 800],
        )

        hdf.create_dataset(
            "s_arrival",
            data=[2228, 2500, 2000],
        )

        hdf.create_dataset(
            "metadata",
            data=np.array(
                ["trace_001", "trace_002", "trace_003"],
                dtype=h5py.string_dtype(),
            ),
        )

    return path


def test_dataset_length(sample_hdf5):
    dataset = CuratedSTEADDataset(sample_hdf5)
    assert len(dataset) == 3


def test_load_record(sample_hdf5):
    dataset = CuratedSTEADDataset(sample_hdf5)
    record = dataset[0]

    assert isinstance(record["waveform"], torch.Tensor)
    assert record["waveform"].shape == (3, 6000)
    assert record["p_arrival"] == 900
    assert record["s_arrival"] == 2228
    assert record["metadata"] == "trace_001"


def test_negative_index(sample_hdf5):
    dataset = CuratedSTEADDataset(sample_hdf5)

    assert dataset[-1]["metadata"] == "trace_003"


def test_out_of_range_index(sample_hdf5):
    dataset = CuratedSTEADDataset(sample_hdf5)

    with pytest.raises(IndexError):
        dataset[3]


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        CuratedSTEADDataset(tmp_path / "missing.hdf5")