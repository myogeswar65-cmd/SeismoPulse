
import h5py
import numpy as np
import pytest
import torch

from seismopulse.data.training_dataset import SeismicPhaseDataset


@pytest.fixture
def sample_files(tmp_path):
    earthquake_path = tmp_path / "earthquake.hdf5"
    noise_path = tmp_path / "noise.hdf5"

    rng = np.random.default_rng(42)

    earthquake_traces = rng.normal(
        0, 100, size=(4, 6000, 3)
    ).astype(np.float32)

    noise_traces = rng.normal(
        0, 50, size=(3, 6000, 3)
    ).astype(np.float32)

    with h5py.File(earthquake_path, "w") as hdf:
        hdf.create_dataset("traces", data=earthquake_traces)
        hdf.create_dataset("p_arrival", data=[900, 950, 1000, 1100])
        hdf.create_dataset("s_arrival", data=[1800, 1900, 2000, 2100])
        hdf.create_dataset(
            "metadata",
            data=np.array(
                ["eq_001", "eq_002", "eq_003", "eq_004"],
                dtype=h5py.string_dtype(),
            ),
        )

    with h5py.File(noise_path, "w") as hdf:
        hdf.create_dataset("traces", data=noise_traces)
        hdf.create_dataset(
            "metadata",
            data=np.array(
                ["noise_001", "noise_002", "noise_003"],
                dtype=h5py.string_dtype(),
            ),
        )

    return earthquake_path, noise_path


@pytest.fixture
def dataset(sample_files):
    earthquake_path, noise_path = sample_files

    return SeismicPhaseDataset(
        earthquake_path=earthquake_path,
        noise_path=noise_path,
        earthquake_indices=[0, 1],
        noise_indices=[0, 1],
    )


def test_dataset_length(dataset):
    assert len(dataset) == 4


def test_earthquake_record(dataset):
    sample = dataset[0]

    assert sample["source"] == "earthquake"
    assert sample["metadata"] == "eq_001"
    assert sample["waveform"].shape == (3, 6000)
    assert sample["targets"].shape == (3, 6000)


def test_noise_record(dataset):
    sample = dataset[2]

    assert sample["source"] == "noise"
    assert sample["metadata"] == "noise_001"
    assert sample["waveform"].shape == (3, 6000)

    assert torch.all(sample["targets"][0] == 1)
    assert torch.all(sample["targets"][1] == 0)
    assert torch.all(sample["targets"][2] == 0)


def test_waveform_normalization(dataset):
    waveform = dataset[0]["waveform"]

    assert torch.isfinite(waveform).all()
    assert torch.allclose(
        waveform.mean(dim=1),
        torch.zeros(3),
        atol=1e-5,
    )
    assert torch.allclose(
        waveform.std(dim=1, unbiased=False),
        torch.ones(3),
        atol=1e-4,
    )


def test_negative_index(dataset):
    assert dataset[-1]["source"] == "noise"


def test_out_of_range_index(dataset):
    with pytest.raises(IndexError):
        dataset[4]


def test_duplicate_indices_rejected(sample_files):
    earthquake_path, noise_path = sample_files

    with pytest.raises(ValueError):
        SeismicPhaseDataset(
            earthquake_path,
            noise_path,
            earthquake_indices=[0, 0],
            noise_indices=[0],
        )


def test_invalid_indices_rejected(sample_files):
    earthquake_path, noise_path = sample_files

    with pytest.raises(IndexError):
        SeismicPhaseDataset(
            earthquake_path,
            noise_path,
            earthquake_indices=[99],
            noise_indices=[0],
        )


def test_non_integer_index_rejected(dataset):
    with pytest.raises(TypeError):
        dataset[1.5]
