
import numpy as np
import pytest

from seismopulse.data.targets import (
    generate_phase_targets,
    NOISE_CLASS,
    P_CLASS,
    S_CLASS,
)


def test_target_shape_and_dtype():
    targets = generate_phase_targets(900, 2228)

    assert targets.shape == (3, 6000)
    assert targets.dtype == np.float32


def test_probability_distribution_sums_to_one():
    targets = generate_phase_targets(900, 2228)

    sums = targets.sum(axis=0)

    assert np.allclose(sums, 1.0, atol=1e-6)


def test_p_arrival_peak():
    targets = generate_phase_targets(900, 2228)

    assert targets[P_CLASS, 900] == pytest.approx(1.0)
    assert targets[P_CLASS, 900] == targets[P_CLASS].max()


def test_s_arrival_peak():
    targets = generate_phase_targets(900, 2228)

    assert targets[S_CLASS, 2228] == pytest.approx(1.0)
    assert targets[S_CLASS, 2228] == targets[S_CLASS].max()


def test_noise_probability_between_arrivals():
    targets = generate_phase_targets(900, 2228)

    assert targets[NOISE_CLASS, 1500] == pytest.approx(1.0)


def test_targets_are_finite_and_nonnegative():
    targets = generate_phase_targets(900, 2228)

    assert np.isfinite(targets).all()
    assert np.all(targets >= 0.0)
    assert np.all(targets <= 1.0)


def test_invalid_phase_order():
    with pytest.raises(ValueError):
        generate_phase_targets(2228, 900)


def test_out_of_range_arrival():
    with pytest.raises(ValueError):
        generate_phase_targets(900, 6000)


def test_invalid_sigma():
    with pytest.raises(ValueError):
        generate_phase_targets(900, 2228, sigma_samples=0)


def test_invalid_sample_count():
    with pytest.raises(ValueError):
        generate_phase_targets(900, 2228, num_samples=0)
