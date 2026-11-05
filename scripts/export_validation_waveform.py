
import sys
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seismopulse.data.build_loaders import build_dataloaders
from seismopulse.models.phase_net import PhaseNet

CHECKPOINT = ROOT / "experiments" / "checkpoints" / "phase_net_best.pt"
OUTPUT = ROOT / "experiments" / "validation_waveform.json"
DEMO_FILE = ROOT / "experiments" / "demo_validation_waveform.json"

SAMPLE_RATE = 100
P_THRESHOLD = 0.30
S_THRESHOLD = 0.30
BATCH_SIZE = 8
TOLERANCE_SAMPLES = 50

# Exclude the previously selected demo record.
EXCLUDED_DATASET_INDICES = {1053}


def main():

    torch.set_num_threads(min(4, torch.get_num_threads()))

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    if not DEMO_FILE.exists():
        raise FileNotFoundError(DEMO_FILE)

    # Load existing demo information.
    with DEMO_FILE.open("r", encoding="utf-8") as f:
        demo = json.load(f)

    demo_metadata = demo.get("metadata")

    demo_waveform = torch.tensor(
        demo["waveform"],
        dtype=torch.float32
    ).T.contiguous()

    print("Existing demo metadata:", demo_metadata)
    print("Excluded dataset indices:", EXCLUDED_DATASET_INDICES)

    print("\nLoading validation dataset...")

    _, val_loader = build_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    dataset = val_loader.dataset
    earthquake_count = len(dataset.earthquake_indices)

    print("Validation earthquake records:", earthquake_count)

    print("\nLoading trained model...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model = PhaseNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    candidates = []

    excluded_count = 0

    print("\nScanning validation earthquakes...")

    with torch.inference_mode():

        for start in range(0, earthquake_count, BATCH_SIZE):

            end = min(start + BATCH_SIZE, earthquake_count)

            items = [
                dataset[i]
                for i in range(start, end)
            ]

            waveforms = torch.stack(
                [item["waveform"] for item in items]
            )

            targets = torch.stack(
                [item["targets"] for item in items]
            )

            # Same normalization as inference.
            means = waveforms.mean(dim=-1, keepdim=True)

            stds = waveforms.std(
                dim=-1,
                keepdim=True,
                unbiased=False,
            )

            normalized = (
                waveforms - means
            ) / stds.clamp(min=1e-6)

            probabilities = torch.softmax(
                model(normalized),
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

                dataset_index = start + j
                metadata = item["metadata"]

                # Exclude the previous demo dataset index.
                if dataset_index in EXCLUDED_DATASET_INDICES:
                    excluded_count += 1
                    continue

                # Exclude the same earthquake ID.
                if metadata == demo_metadata:
                    excluded_count += 1
                    continue

                # Exclude identical waveforms, even if metadata differs.
                original_waveform = item["waveform"].detach().cpu()

                if original_waveform.shape == demo_waveform.shape:

                    if torch.allclose(
                        original_waveform,
                        demo_waveform,
                        rtol=1e-5,
                        atol=1e-6,
                    ):
                        excluded_count += 1
                        continue

                true_p = int(
                    torch.argmax(targets[j, 1]).item()
                )

                true_s = int(
                    torch.argmax(targets[j, 2]).item()
                )

                pred_p = int(p_samples[j].item())
                pred_s = int(s_samples[j].item())

                p_confidence = float(p_conf[j].item())
                s_confidence = float(s_conf[j].item())

                p_detected = p_confidence >= P_THRESHOLD
                s_detected = s_confidence >= S_THRESHOLD

                if (
                    p_detected
                    and s_detected
                    and pred_s <= pred_p
                ):
                    s_detected = False

                p_error = abs(pred_p - true_p)
                s_error = abs(pred_s - true_s)

                candidates.append({

                    "dataset_index": dataset_index,

                    "waveform": item["waveform"].numpy().T.tolist(),

                    "metadata": metadata,

                    "true_p": true_p,
                    "true_s": true_s,

                    "pred_p": pred_p,
                    "pred_s": pred_s,

                    "p_confidence": p_confidence,
                    "s_confidence": s_confidence,

                    "p_detected": p_detected,
                    "s_detected": s_detected,

                    "p_error": p_error,
                    "s_error": s_error,

                })

            if end % 400 < BATCH_SIZE or end == earthquake_count:

                print(f"Scanned {end}/{earthquake_count}")

    if not candidates:
        raise RuntimeError(
            "No alternative validation records found."
        )

    # Prefer both phases detected, then accurate timing.
    candidates.sort(
        key=lambda x: (
            not (x["p_detected"] and x["s_detected"]),
            x["p_error"] + x["s_error"],
            -min(x["p_confidence"], x["s_confidence"]),
        )
    )

    best = candidates[0]

    p_time = round(best["pred_p"] / SAMPLE_RATE, 4)
    s_time = round(best["pred_s"] / SAMPLE_RATE, 4)

    both_detected = (
        best["p_detected"] and best["s_detected"]
    )

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

            "status": (
                "phases_detected"
                if both_detected
                else "no_confident_phase"
            ),

            "sample_rate_hz": SAMPLE_RATE,

            "p_wave": {
                "detected": best["p_detected"],
                "sample": best["pred_p"] if best["p_detected"] else None,
                "time_seconds": p_time if best["p_detected"] else None,
                "confidence": round(best["p_confidence"], 5),
                "threshold": P_THRESHOLD,
            },

            "s_wave": {
                "detected": best["s_detected"],
                "sample": best["pred_s"] if best["s_detected"] else None,
                "time_seconds": s_time if best["s_detected"] else None,
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

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n========== NEW VALIDATION RECORD ==========")

    print("Excluded records:", excluded_count)
    print("Dataset index:", best["dataset_index"])
    print("Metadata:", best["metadata"])

    print("P detected:", best["p_detected"])
    print("S detected:", best["s_detected"])

    print("P confidence:", f"{best['p_confidence']:.2%}")
    print("S confidence:", f"{best['s_confidence']:.2%}")

    print("Ground-truth P:", best["true_p"])
    print("Ground-truth S:", best["true_s"])

    print("Predicted P:", best["pred_p"])
    print("Predicted S:", best["pred_s"])

    print("P error:", best["p_error"] * 10, "ms")
    print("S error:", best["s_error"] * 10, "ms")

    print(
        "Both within 500ms:",
        result["evaluation"]["within_500ms"]
    )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
