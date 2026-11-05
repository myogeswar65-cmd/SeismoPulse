
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from seismopulse.data.build_loaders import build_dataloaders
from seismopulse.models.phase_net import PhaseNet


CHECKPOINT = ROOT / "experiments" / "checkpoints" / "phase_net_best.pt"
OUTPUT = ROOT / "experiments" / "demo_validation_waveform.json"

SAMPLE_RATE = 100
P_THRESHOLD = 0.30
S_THRESHOLD = 0.30
BATCH_SIZE = 8
TOLERANCE_SAMPLES = 50  # 500 ms at 100 Hz


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    print("Loading validation dataset...")

    _, val_loader = build_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    dataset = val_loader.dataset
    earthquake_count = len(dataset.earthquake_indices)

    print(f"Validation earthquake records: {earthquake_count}")
    print("Final test dataset: NOT LOADED")
    print("Loading trained model...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model = PhaseNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    candidates = []

    print("\nScanning validation earthquake records...")

    with torch.inference_mode():
        for start in range(0, earthquake_count, BATCH_SIZE):
            end = min(start + BATCH_SIZE, earthquake_count)

            items = [dataset[i] for i in range(start, end)]

            waveforms = torch.stack(
                [item["waveform"] for item in items]
            )

            targets = torch.stack(
                [item["targets"] for item in items]
            )

            # Match the normalization used by predict_phases().
            means = waveforms.mean(dim=-1, keepdim=True)
            stds = waveforms.std(
                dim=-1,
                keepdim=True,
                unbiased=False,
            )

            waveforms = (
                waveforms - means
            ) / stds.clamp(min=1e-6)

            probabilities = torch.softmax(
                model(waveforms),
                dim=1,
            )

            p_probs = probabilities[:, 1, :].clone()
            s_probs = probabilities[:, 2, :].clone()

            edge = min(50, p_probs.shape[-1] // 4)

            if edge > 0:
                p_probs[:, :edge] = 0
                p_probs[:, -edge:] = 0
                s_probs[:, :edge] = 0
                s_probs[:, -edge:] = 0

            p_conf, p_samples = p_probs.max(dim=1)
            s_conf, s_samples = s_probs.max(dim=1)

            for j, item in enumerate(items):
                p_sample = int(p_samples[j].item())
                s_sample = int(s_samples[j].item())

                p_confidence = float(p_conf[j].item())
                s_confidence = float(s_conf[j].item())

                p_detected = p_confidence >= P_THRESHOLD
                s_detected = s_confidence >= S_THRESHOLD

                # Apply the same P-before-S rule as phase_picker.py.
                if p_detected and s_detected and s_sample <= p_sample:
                    s_detected = False

                if not (p_detected and s_detected):
                    continue

                # Ground-truth arrival samples from generated targets.
                true_p = int(torch.argmax(targets[j, 1]).item())
                true_s = int(torch.argmax(targets[j, 2]).item())

                p_error = abs(p_sample - true_p)
                s_error = abs(s_sample - true_s)

                candidates.append({
                    "dataset_index": start + j,
                    "waveform": item["waveform"].numpy().T.tolist(),
                    "metadata": item["metadata"],
                    "true_p": true_p,
                    "true_s": true_s,
                    "pred_p": p_sample,
                    "pred_s": s_sample,
                    "p_confidence": p_confidence,
                    "s_confidence": s_confidence,
                    "p_error": p_error,
                    "s_error": s_error,
                })

            if end % 400 < BATCH_SIZE or end == earthquake_count:
                print(
                    f"Scanned {end}/{earthquake_count} | "
                    f"Candidates with both phases: {len(candidates)}"
                )

    if not candidates:
        print("\nNo validation record had both phases detected.")
        print("No demo file was created.")
        return

    # Prefer records where both predictions are within 500 ms
    # of their ground-truth arrivals, then minimize total error.
    candidates.sort(
        key=lambda x: (
            x["p_error"] > TOLERANCE_SAMPLES
            or x["s_error"] > TOLERANCE_SAMPLES,
            x["p_error"] + x["s_error"],
            -min(x["p_confidence"], x["s_confidence"]),
        )
    )

    best = candidates[0]

    p_time = round(best["pred_p"] / SAMPLE_RATE, 4)
    s_time = round(best["pred_s"] / SAMPLE_RATE, 4)

    result = {
        "model": "SeismoPulse PhaseNet-inspired",
        "source": "validation_earthquake",
        "sample_rate_hz": SAMPLE_RATE,
        "waveform": best["waveform"],
        "metadata": best["metadata"],
        "ground_truth": {
            "p_arrival_sample": best["true_p"],
            "s_arrival_sample": best["true_s"],
        },
        "prediction": {
            "status": "phases_detected",
            "sample_rate_hz": SAMPLE_RATE,
            "p_wave": {
                "detected": True,
                "sample": best["pred_p"],
                "time_seconds": p_time,
                "confidence": round(best["p_confidence"], 5),
                "threshold": P_THRESHOLD,
            },
            "s_wave": {
                "detected": True,
                "sample": best["pred_s"],
                "time_seconds": s_time,
                "confidence": round(best["s_confidence"], 5),
                "threshold": S_THRESHOLD,
            },
        },
        "evaluation": {
            "p_error_ms": best["p_error"] * 10,
            "s_error_ms": best["s_error"] * 10,
            "within_500ms": (
                best["p_error"] <= TOLERANCE_SAMPLES
                and best["s_error"] <= TOLERANCE_SAMPLES
            ),
        },
        "validation_dataset_index": best["dataset_index"],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print("\n========== SELECTED DEMO WAVEFORM ==========")
    print(f"Validation index: {best['dataset_index']}")
    print(f"P confidence:     {best['p_confidence']:.2%}")
    print(f"S confidence:     {best['s_confidence']:.2%}")
    print(f"P arrival:        {p_time:.3f} seconds")
    print(f"S arrival:        {s_time:.3f} seconds")
    print(f"P timing error:   {best['p_error'] * 10} ms")
    print(f"S timing error:   {best['s_error'] * 10} ms")
    print(f"Both within 500ms: {result['evaluation']['within_500ms']}")
    print(f"\nSaved: {OUTPUT}")
    print("============================================")


if __name__ == "__main__":
    main()
