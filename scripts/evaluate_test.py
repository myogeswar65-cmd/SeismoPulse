
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.seismopulse.data.curated_stead import CuratedSTEADDataset
from src.seismopulse.models.phase_net import PhaseNet

# Configuration
CHECKPOINT = ROOT / "experiments/checkpoints/phase_net_best.pt"
EARTHQUAKE_TEST = ROOT / "data/raw/stead/test.hdf5"
NOISE_TEST = ROOT / "data/raw/stead/test_noise.hdf5"
OUTPUT = ROOT / "experiments/test_evaluation.json"

SAMPLE_RATE = 100
BATCH_SIZE = 4
TOLERANCE_SAMPLES = 50

P_THRESHOLD = 0.30
S_THRESHOLD = 0.30
EDGE_SAMPLES = 50


def load_model():
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    model = PhaseNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\n========== CHECKPOINT ==========")
    print("Epoch:", checkpoint["epoch"])
    print("Validation loss:", checkpoint["validation_loss"])
    print("P threshold:", P_THRESHOLD)
    print("S threshold:", S_THRESHOLD)

    return model


def normalize_waveforms(waveforms):
    mean = waveforms.mean(dim=-1, keepdim=True)
    std = waveforms.std(
        dim=-1,
        keepdim=True,
        unbiased=False
    )
    return (waveforms - mean) / std.clamp(min=1e-6)


def find_arrival(probability):
    probability = probability.clone()

    edge = min(EDGE_SAMPLES, probability.numel() // 4)

    if edge:
        probability[:edge] = 0
        probability[-edge:] = 0

    index = int(torch.argmax(probability).item())
    confidence = float(probability[index].item())

    return index, confidence


def phase_statistics(errors, accepted, total):
    accepted_count = len(errors)

    if accepted_count:
        values = np.asarray(errors, dtype=np.float64)

        timing = {
            "mean_error_ms": float(np.mean(values) * 10),
            "median_error_ms": float(np.median(values) * 10),
            "p90_error_ms": float(np.percentile(values, 90) * 10),
            "within_500ms_percent": float(
                100 * np.mean(values <= TOLERANCE_SAMPLES)
            )
        }
    else:
        timing = {
            "mean_error_ms": None,
            "median_error_ms": None,
            "p90_error_ms": None,
            "within_500ms_percent": None
        }

    return {
        "total_records": total,
        "accepted_predictions": accepted,
        "rejected_predictions": total - accepted,
        "coverage_percent": (
            100 * accepted / total if total else None
        ),
        "timing_evaluated_predictions": accepted_count,
        "timing": timing
    }


def evaluate_earthquakes(model):
    dataset = CuratedSTEADDataset(str(EARTHQUAKE_TEST))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    p_errors = []
    s_errors = []

    p_accepted = 0
    s_accepted = 0
    valid_records = 0
    skipped_records = 0
    ordered_accepted = 0

    print("\n========== EARTHQUAKE TEST ==========")

    with torch.no_grad():
        for batch_number, batch in enumerate(loader, start=1):

            waveforms = normalize_waveforms(
                batch["waveform"].float()
            )

            probabilities = torch.softmax(
                model(waveforms),
                dim=1
            )

            for i in range(len(waveforms)):

                p_true = int(batch["p_arrival"][i])
                s_true = int(batch["s_arrival"][i])

                if (
                    p_true < EDGE_SAMPLES
                    or s_true < EDGE_SAMPLES
                    or p_true >= 5950
                    or s_true >= 5950
                    or p_true >= s_true
                ):
                    skipped_records += 1
                    continue

                p_pred, p_conf = find_arrival(
                    probabilities[i, 1]
                )

                s_pred, s_conf = find_arrival(
                    probabilities[i, 2]
                )

                valid_records += 1

                p_detected = p_conf >= P_THRESHOLD
                s_detected = s_conf >= S_THRESHOLD

                # Enforce P-before-S ordering when both are accepted.
                if (
                    p_detected
                    and s_detected
                    and p_pred >= s_pred
                ):
                    s_detected = False

                if p_detected:
                    p_accepted += 1
                    p_errors.append(abs(p_pred - p_true))

                if s_detected:
                    s_accepted += 1
                    s_errors.append(abs(s_pred - s_true))

                if p_detected and s_detected and p_pred < s_pred:
                    ordered_accepted += 1

            if batch_number % 100 == 0:
                print(
                    f"Processed "
                    f"{min(batch_number * BATCH_SIZE, len(dataset))}"
                    f"/{len(dataset)} earthquake records"
                )

    return {
        "records_evaluated": valid_records,
        "records_skipped": skipped_records,
        "thresholds": {
            "p_wave": P_THRESHOLD,
            "s_wave": S_THRESHOLD
        },
        "p_wave": phase_statistics(
            p_errors, p_accepted, valid_records
        ),
        "s_wave": phase_statistics(
            s_errors, s_accepted, valid_records
        ),
        "both_phases_detected_and_ordered_percent": (
            100 * ordered_accepted / valid_records
            if valid_records else None
        )
    }


def evaluate_noise(model):
    noise_correct = 0
    total_samples = 0

    records_with_p = 0
    records_with_s = 0
    records_with_any_phase = 0

    records_processed = 0

    print("\n========== NOISE TEST ==========")

    with h5py.File(NOISE_TEST, "r") as hdf:

        if "traces" not in hdf:
            raise ValueError("Noise HDF5 file does not contain traces.")

        traces = hdf["traces"]

        if traces.ndim != 3 or traces.shape[-1] != 3:
            raise ValueError(
                f"Unexpected noise waveform shape: {traces.shape}"
            )

        total_records = traces.shape[0]

        with torch.no_grad():

            for start in range(0, total_records, BATCH_SIZE):

                end = min(start + BATCH_SIZE, total_records)

                waveforms = torch.from_numpy(
                    np.asarray(
                        traces[start:end],
                        dtype=np.float32
                    )
                ).permute(0, 2, 1).contiguous()

                waveforms = normalize_waveforms(waveforms)

                probabilities = torch.softmax(
                    model(waveforms),
                    dim=1
                )

                predictions = probabilities.argmax(dim=1)

                noise_correct += int(
                    (predictions == 0).sum().item()
                )
                total_samples += predictions.numel()

                for i in range(len(waveforms)):

                    p_pred, p_conf = find_arrival(
                        probabilities[i, 1]
                    )

                    s_pred, s_conf = find_arrival(
                        probabilities[i, 2]
                    )

                    has_p = p_conf >= P_THRESHOLD
                    has_s = s_conf >= S_THRESHOLD

                    records_with_p += int(has_p)
                    records_with_s += int(has_s)
                    records_with_any_phase += int(has_p or has_s)

                records_processed += len(waveforms)

                if (
                    records_processed % 400 == 0
                    or records_processed == total_records
                ):
                    print(
                        f"Processed {records_processed}/"
                        f"{total_records} noise records"
                    )

    return {
        "records_evaluated": records_processed,
        "noise_pixel_accuracy_percent": (
            100 * noise_correct / total_samples
            if total_samples else None
        ),
        "records_with_p_predictions": records_with_p,
        "records_with_s_predictions": records_with_s,
        "records_with_any_predicted_phase": records_with_any_phase,
        "false_phase_record_percent": (
            100 * records_with_any_phase / records_processed
            if records_processed else None
        ),
        "thresholds": {
            "p_wave": P_THRESHOLD,
            "s_wave": S_THRESHOLD
        }
    }


def main():
    print("\n==============================================")
    print("       SEISMOPULSE FINAL TEST EVALUATION")
    print("==============================================")

    for path in [CHECKPOINT, EARTHQUAKE_TEST, NOISE_TEST]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    torch.set_num_threads(min(4, torch.get_num_threads()))

    model = load_model()

    earthquake_results = evaluate_earthquakes(model)
    noise_results = evaluate_noise(model)

    results = {
        "model": "SeismoPulse PhaseNet-inspired model",
        "checkpoint_epoch": 5,
        "device": "cpu",
        "sample_rate_hz": SAMPLE_RATE,
        "arrival_tolerance_ms": 500,
        "thresholds_calibrated_on_validation": {
            "p_wave": P_THRESHOLD,
            "s_wave": S_THRESHOLD
        },
        "earthquake_test": earthquake_results,
        "noise_test": noise_results,
        "evaluation_note": (
            "Timing accuracy is calculated only for accepted "
            "predictions. Coverage is reported separately. "
            "Thresholds were calibrated on validation data."
        )
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8"
    )

    print("\n========== FINAL TEST RESULTS ==========")
    print(json.dumps(results, indent=4))
    print("\nSaved:", OUTPUT)
    print("\nTEST EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
