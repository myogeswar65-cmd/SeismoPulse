
import numpy as np
import pytest

from seismopulse.data.validation import validate_waveform


def test_valid_waveform():
    waveform = np.random.randn(3, 6000)

    # Should accept a valid 3-component waveform.
    validate_waveform(waveform, sampling_rate=100.0)


def test_rejects_wrong_number_of_components():
    waveform = np.random.randn(2, 6000)

    with pytest.raises(ValueError):
        validate_waveform(waveform, sampling_rate=100.0)


def test_rejects_empty_waveform():
    waveform = np.empty((3, 0))

    with pytest.raises(ValueError):
        validate_waveform(waveform, sampling_rate=100.0)


def test_rejects_nan_values():
    waveform = np.random.randn(3, 6000)
    waveform[0, 100] = np.nan

    with pytest.raises(ValueError):
        validate_waveform(waveform, sampling_rate=100.0)


def test_rejects_invalid_sampling_rate():
    waveform = np.random.randn(3, 6000)

    with pytest.raises(ValueError):
        validate_waveform(waveform, sampling_rate=0)


def test_rejects_infinite_values():
    waveform = np.random.randn(3, 6000)
    waveform[1, 200] = np.inf

    with pytest.raises(ValueError):
        validate_waveform(waveform, sampling_rate=100.0)
def test_rejects_complex_waveform():
    waveform = np.ones((3, 6000), dtype=np.complex64)

    with pytest.raises(ValueError):
        validate_waveform(waveform, sampling_rate=100.0)


def test_rejects_all_zero_waveform():
    waveform = np.zeros((3, 6000))

    with pytest.raises(ValueError):
        validate_waveform(waveform, sampling_rate=100.0)


def test_rejects_boolean_sampling_rate():
    waveform = np.random.randn(3, 6000)

    with pytest.raises(ValueError):
        validate_waveform(waveform, sampling_rate=True)