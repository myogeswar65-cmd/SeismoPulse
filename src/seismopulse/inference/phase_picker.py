
import numpy as np
import torch


DEFAULT_P_THRESHOLD = 0.30
DEFAULT_S_THRESHOLD = 0.30
EDGE_SAMPLES = 50


def normalize_waveform(waveform):
    """
    Normalize a three-component seismic waveform.

    Accepted shapes:
        (3, samples)
        (samples, 3)

    Returns:
        Normalized tensor with shape (3, samples).
    """

    waveform = torch.as_tensor(
        waveform,
        dtype=torch.float32
    )

    if waveform.ndim != 2:
        raise ValueError("Waveform must be a 2D array.")

    # Accept (samples, 3) or (3, samples).
    if waveform.shape[0] == 3:
        pass

    elif waveform.shape[1] == 3:
        waveform = waveform.transpose(0, 1)

    else:
        raise ValueError(
            "Waveform must have exactly 3 components."
        )

    if waveform.shape[1] == 0:
        raise ValueError("Waveform cannot be empty.")

    if not torch.isfinite(waveform).all():
        raise ValueError(
            "Waveform contains NaN or infinite values."
        )

    # Per-component normalization.
    mean = waveform.mean(
        dim=-1,
        keepdim=True
    )

    std = waveform.std(
        dim=-1,
        keepdim=True,
        unbiased=False
    )

    normalized = (
        waveform - mean
    ) / std.clamp(min=1e-6)

    return normalized


def predict_phases(
    waveform,
    model,
    sample_rate_hz=100,
    p_threshold=DEFAULT_P_THRESHOLD,
    s_threshold=DEFAULT_S_THRESHOLD
):
    """
    Predict P-wave and S-wave arrivals.

    The model produces three probability classes:
        0 = Noise
        1 = P-wave
        2 = S-wave

    Detection thresholds remain calibrated values.
    S-wave detections must follow P-wave detections
    when both phases are detected.
    """

    # --------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------

    if sample_rate_hz <= 0:
        raise ValueError(
            "Sample rate must be positive."
        )

    if not 0 <= p_threshold <= 1:
        raise ValueError(
            "P-wave threshold must be between 0 and 1."
        )

    if not 0 <= s_threshold <= 1:
        raise ValueError(
            "S-wave threshold must be between 0 and 1."
        )

    # --------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------

    normalized = normalize_waveform(waveform)

    device = next(model.parameters()).device

    normalized = normalized.unsqueeze(0).to(device)

    # --------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------

    model.eval()

    with torch.no_grad():

        logits = model(normalized)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    # --------------------------------------------------
    # P-WAVE AND S-WAVE PROBABILITIES
    # --------------------------------------------------

    p_prob = probabilities[1].clone()

    s_prob = probabilities[2].clone()

    # --------------------------------------------------
    # REMOVE EDGE PREDICTIONS
    # --------------------------------------------------

    edge = min(
        EDGE_SAMPLES,
        p_prob.numel() // 4
    )

    if edge > 0:

        p_prob[:edge] = 0
        p_prob[-edge:] = 0

        s_prob[:edge] = 0
        s_prob[-edge:] = 0

    # --------------------------------------------------
    # FIND PEAK PROBABILITIES
    # --------------------------------------------------

    p_sample = int(
        torch.argmax(p_prob).item()
    )

    s_sample = int(
        torch.argmax(s_prob).item()
    )

    p_confidence = float(
        p_prob[p_sample].item()
    )

    s_confidence = float(
        s_prob[s_sample].item()
    )

    # --------------------------------------------------
    # THRESHOLD-BASED DETECTION
    # --------------------------------------------------

    p_detected = (
        p_confidence >= p_threshold
    )

    s_detected = (
        s_confidence >= s_threshold
    )

    # --------------------------------------------------
    # P-WAVE ARRIVAL
    # --------------------------------------------------

    if p_detected:

        p_sample_result = p_sample

        p_time = round(
            p_sample / sample_rate_hz,
            4
        )

    else:

        p_sample_result = None

        p_time = None

    # --------------------------------------------------
    # S-WAVE ARRIVAL
    # --------------------------------------------------

    if not s_detected:

        s_sample_result = None

        s_time = None

    elif p_detected and s_sample <= p_sample:

        # Reject an S-wave pick that violates
        # P-before-S ordering.

        s_sample_result = None

        s_time = None

        s_detected = False

    else:

        s_sample_result = s_sample

        s_time = round(
            s_sample / sample_rate_hz,
            4
        )

    # --------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------

    if p_detected or s_detected:

        status = "phases_detected"

    else:

        status = "no_confident_phase"

    # --------------------------------------------------
    # RESPONSE
    # --------------------------------------------------

    return {

        "status": status,

        "sample_rate_hz": sample_rate_hz,

        "p_wave": {

            "detected": p_detected,

            "sample": p_sample_result,

            "time_seconds": p_time,

            "confidence": round(
                p_confidence,
                5
            ),

            "threshold": p_threshold

        },

        "s_wave": {

            "detected": s_detected,

            "sample": s_sample_result,

            "time_seconds": s_time,

            "confidence": round(
                s_confidence,
                5
            ),

            "threshold": s_threshold

        }

    }
