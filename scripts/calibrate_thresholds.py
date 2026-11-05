
import sys
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.seismopulse.data.build_loaders import build_dataloaders
from src.seismopulse.models.phase_net import PhaseNet


CHECKPOINT = ROOT / "experiments/checkpoints/phase_net_best.pt"
OUTPUT = ROOT / "experiments/calibrated_thresholds.json"

BATCH_SIZE = 4
TOLERANCE = 50  # 500 ms at 100 Hz
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

MIN_COVERAGE = 70.0
MAX_NOISE_FALSE_ALARM = 5.0


def normalize_waveforms(waveforms):
    mean = waveforms.mean(dim=-1, keepdim=True)
    std = waveforms.std(dim=-1, keepdim=True, unbiased=False)
    return (waveforms - mean) / std.clamp(min=1e-6)


def load_model():
    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    model = PhaseNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Loaded checkpoint epoch:", checkpoint["epoch"])
    return model


def collect_predictions(model, loader):
    earthquakes = []
    noise_records = []

    print("\nCollecting earthquake and noise validation predictions...")

    with torch.no_grad():
        for batch_number, batch in enumerate(loader, start=1):
            waveforms = normalize_waveforms(batch["waveform"].float())
            probabilities = torch.softmax(model(waveforms), dim=1)

            for i in range(len(waveforms)):
                p_prob = probabilities[i, 1].clone()
                s_prob = probabilities[i, 2].clone()

                p_prob[:50] = 0
                p_prob[-50:] = 0
                s_prob[:50] = 0
                s_prob[-50:] = 0

                p_pred = int(p_prob.argmax())
                s_pred = int(s_prob.argmax())

                record = {
                    "p_conf": float(p_prob[p_pred]),
                    "s_conf": float(s_prob[s_pred])
                }

                if batch["source"][i] == "earthquake":
                    targets = batch["targets"][i]
                    p_region = torch.where(targets[1] >= 0.5)[0]
                    s_region = torch.where(targets[2] >= 0.5)[0]

                    if len(p_region) == 0 or len(s_region) == 0:
                        continue

                    p_true = int(p_region[len(p_region) // 2])
                    s_true = int(s_region[len(s_region) // 2])

                    if p_true >= s_true:
                        continue

                    record.update({
                        "p_true": p_true,
                        "s_true": s_true,
                        "p_pred": p_pred,
                        "s_pred": s_pred
                    })
                    earthquakes.append(record)

                elif batch["source"][i] == "noise":
                    noise_records.append(record)

            if batch_number % 200 == 0:
                print(f"Processed {batch_number} batches")

    print("Earthquake validation records:", len(earthquakes))
    print("Noise validation records:", len(noise_records))

    return earthquakes, noise_records


def evaluate_threshold(earthquakes, noise_records, phase, threshold):
    conf_key = f"{phase}_conf"
    true_key = f"{phase}_true"
    pred_key = f"{phase}_pred"

    accepted = [
        r for r in earthquakes if r[conf_key] >= threshold
    ]

    noise_alarms = [
        r for r in noise_records if r[conf_key] >= threshold
    ]

    if accepted:
        errors = np.array([
            abs(r[pred_key] - r[true_key]) for r in accepted
        ])

        accuracy = float(100 * np.mean(errors <= TOLERANCE))
        median_error = float(np.median(errors) * 10)
    else:
        accuracy = None
        median_error = None

    coverage = (
        100 * len(accepted) / len(earthquakes)
        if earthquakes else 0.0
    )

    false_alarm = (
        100 * len(noise_alarms) / len(noise_records)
        if noise_records else 0.0
    )

    return {
        "threshold": threshold,
        "coverage_percent": round(coverage, 2),
        "accuracy_within_500ms_percent": (
            round(accuracy, 2) if accuracy is not None else None
        ),
        "median_error_ms": median_error,
        "accepted_earthquake_records": len(accepted),
        "noise_false_alarm_percent": round(false_alarm, 2),
        "noise_records_with_alarm": len(noise_alarms)
    }


def choose_threshold(results):
    eligible = [
        r for r in results
        if r["coverage_percent"] >= MIN_COVERAGE
        and r["noise_false_alarm_percent"] <= MAX_NOISE_FALSE_ALARM
        and r["accuracy_within_500ms_percent"] is not None
    ]

    if eligible:
        return max(
            eligible,
            key=lambda r: r["accuracy_within_500ms_percent"]
        )

    # If no threshold meets both constraints, prefer lower noise
    # false alarms while retaining the required earthquake coverage.
    coverage_candidates = [
        r for r in results
        if r["coverage_percent"] >= MIN_COVERAGE
    ]

    if coverage_candidates:
        return min(
            coverage_candidates,
            key=lambda r: (
                r["noise_false_alarm_percent"],
                -(r["accuracy_within_500ms_percent"] or 0)
            )
        )

    return max(
        results,
        key=lambda r: r["accuracy_within_500ms_percent"] or 0
    )


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    model = load_model()

    _, validation_loader = build_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    earthquakes, noise_records = collect_predictions(
        model, validation_loader
    )

    if not earthquakes or not noise_records:
        raise RuntimeError(
            "Missing earthquake or noise validation records."
        )

    results = {}

    for phase in ["p", "s"]:
        threshold_results = [
            evaluate_threshold(
                earthquakes, noise_records, phase, threshold
            )
            for threshold in THRESHOLDS
        ]

        results[phase] = {
            "threshold_results": threshold_results,
            "recommended_threshold": choose_threshold(threshold_results)
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8"
    )

    print("\n========== FINAL CALIBRATION ==========")

    for phase in ["p", "s"]:
        print(f"\n{phase.upper()}-WAVE")

        for r in results[phase]["threshold_results"]:
            print(
                f"Threshold: {r['threshold']:.2f} | "
                f"Coverage: {r['coverage_percent']:.2f}% | "
                f"Accuracy: {r['accuracy_within_500ms_percent']}% | "
                f"Noise false alarms: {r['noise_false_alarm_percent']:.2f}%"
            )

        print("\nRecommended:", results[phase]["recommended_threshold"])

    print("\nSaved:", OUTPUT)
    print("\nVALIDATION CALIBRATION COMPLETED")


if __name__ == "__main__":
    main()
