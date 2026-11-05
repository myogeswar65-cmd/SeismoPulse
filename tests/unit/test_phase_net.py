import torch

from seismopulse.models.phase_net import PhaseNet


def test_model_output_shape():
    model = PhaseNet()

    waveform = torch.randn(2, 3, 6000)

    output = model(waveform)

    assert output.shape == (2, 3, 6000)


def test_model_produces_finite_values():
    model = PhaseNet()

    waveform = torch.randn(1, 3, 6000)

    output = model(waveform)

    assert torch.isfinite(output).all()


def test_model_probability_distribution():
    model = PhaseNet()

    waveform = torch.randn(1, 3, 6000)

    logits = model(waveform)

    probabilities = torch.softmax(logits, dim=1)

    assert probabilities.shape == (1, 3, 6000)

    torch.testing.assert_close(
        probabilities.sum(dim=1),
        torch.ones(1, 6000),
    )