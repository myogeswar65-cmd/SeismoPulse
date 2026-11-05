import h5py
import numpy as np
import pytest

from seismopulse.data.waveform_reader import load_waveform


@pytest.fixture
def sample_hdf5(tmp_path):
    file_path = tmp_path / "sample.hdf5"

    waveform = np.random.default_rng(42).normal(
        size=(6000, 3)
    ).astype(np.float32)

    with h5py.File(file_path, "w") as f:
        group = f.create_group("data")
        group.create_dataset("trace_001", data=waveform)

    return file_path, waveform


def test_load_waveform_successfully(sample_hdf5):
    file_path, original = sample_hdf5

    result = load_waveform(file_path, "trace_001")

    assert result.shape == (3, 6000)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, original.T)


def test_load_waveform_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_waveform(tmp_path / "missing.hdf5", "trace_001")


def test_load_waveform_trace_not_found(sample_hdf5):
    file_path, _ = sample_hdf5

    with pytest.raises(KeyError):
        load_waveform(file_path, "unknown_trace")


def test_load_waveform_rejects_invalid_shape(tmp_path):
    file_path = tmp_path / "invalid.hdf5"

    with h5py.File(file_path, "w") as f:
        group = f.create_group("data")
        group.create_dataset(
            "trace_001",
            data=np.ones((6000, 2)),
        )

    with pytest.raises(ValueError):
        load_waveform(file_path, "trace_001")