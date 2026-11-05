
"""SeismoPulse: Train and evaluate a seismic phase-picking model."""

import argparse
import json
import random
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

from seismopulse.data.build_loaders import build_dataloaders
from seismopulse.models.phase_net import PhaseNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = ["Noise", "P-wave", "S-wave"]
NUM_CLASSES = 3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def phase_loss(logits, targets):
    """
    Hybrid soft cross-entropy:
    - Standard loss preserves general classification.
    - Class-weighted loss increases phase contribution.
    """

    log_probabilities = F.log_softmax(logits, dim=1)

    pixel_loss = -(targets * log_probabilities)

    standard_loss = pixel_loss.sum(dim=1).mean()

    class_mass = targets.sum(dim=(0, 2)).clamp_min(1.0)

    class_weights = torch.sqrt(
        class_mass.sum() / class_mass
    )

    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.clamp(max=5.0)

    weighted_pixel_loss = (
        pixel_loss * class_weights.view(1, NUM_CLASSES, 1)
    )

    balanced_loss = weighted_pixel_loss.sum(dim=1).mean()

    return 0.75 * standard_loss + 0.25 * balanced_loss


def update_confusion_matrix(confusion, predictions, targets):
    """
    Evaluate phase predictions using Gaussian arrival regions.

    Gaussian target values >= 0.5 define the central phase region.
    Remaining samples are treated as Noise.
    """

    predicted = predictions.argmax(dim=1)

    actual = torch.zeros_like(predicted)

    p_region = targets[:, 1] >= 0.5
    s_region = targets[:, 2] >= 0.5

    actual[p_region] = 1
    actual[s_region] = 2

    predicted = predicted.reshape(-1)
    actual = actual.reshape(-1)

    encoded = actual * NUM_CLASSES + predicted

    counts = torch.bincount(
        encoded,
        minlength=NUM_CLASSES * NUM_CLASSES
    )

    confusion += counts.reshape(
        NUM_CLASSES,
        NUM_CLASSES
    ).cpu()


def calculate_metrics(confusion):
    """Calculate class-wise IoU, precision, recall and F1."""

    metrics = {}

    for index, name in enumerate(CLASS_NAMES):

        tp = confusion[index, index].item()

        fp = confusion[:, index].sum().item() - tp
        fn = confusion[index, :].sum().item() - tp

        iou_denominator = tp + fp + fn
        precision_denominator = tp + fp
        recall_denominator = tp + fn

        iou = (
            tp / iou_denominator
            if iou_denominator
            else 0.0
        )

        precision = (
            tp / precision_denominator
            if precision_denominator
            else 0.0
        )

        recall = (
            tp / recall_denominator
            if recall_denominator
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        metrics[name] = {
            "iou": round(iou, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return metrics


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    max_batches=None
):
    """Run one training or validation epoch."""

    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    total_batches = 0

    confusion = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=torch.int64
    )

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:

        for batch_index, batch in enumerate(loader):

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

            waveforms = batch["waveform"].to(device)
            targets = batch["targets"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(waveforms)

            loss = phase_loss(logits, targets)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at batch {batch_index}."
                )

            if training:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            update_confusion_matrix(
                confusion,
                logits.detach(),
                targets
            )

    if total_batches == 0:
        raise RuntimeError("No batches were processed.")

    return {
        "loss": total_loss / total_batches,
        "metrics": calculate_metrics(confusion),
        "confusion_matrix": confusion.tolist(),
        "batches": total_batches,
    }


def load_history(history_path):
    """Preserve results from previous experiments."""

    if not history_path.exists():
        return []

    try:
        with history_path.open("r", encoding="utf-8") as file:
            previous = json.load(file)

        return previous if isinstance(previous, list) else []

    except (json.JSONDecodeError, OSError):
        print("WARNING: Existing history could not be read.")
        return []


def main():

    parser = argparse.ArgumentParser(
        description="Train the SeismoPulse phase-picking model."
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto"
    )

    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)

    parser.add_argument("--split-path", type=str, default=None)
    parser.add_argument("--earthquake-path", type=str, default=None)
    parser.add_argument("--noise-path", type=str, default=None)

    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/checkpoints/phase_net_best.pt"
    )

    parser.add_argument("--no-checkpoint", action="store_true")

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive.")

    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive.")

    if args.num_workers < 0:
        parser.error("num-workers cannot be negative.")

    if (
        args.max_train_batches is not None
        and args.max_train_batches < 1
    ):
        parser.error("max-train-batches must be positive.")

    if (
        args.max_val_batches is not None
        and args.max_val_batches < 1
    ):
        parser.error("max-val-batches must be positive.")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested, but CUDA is unavailable.")

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if (
            args.device == "cuda"
            or (
                args.device == "auto"
                and torch.cuda.is_available()
            )
        )
        else "cpu"
    )

    # Limit excessive CPU thread overhead on local machines.
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    print("\n========== SEISMOPULSE TRAINING ==========")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("Loss: Hybrid standard + class-weighted soft cross-entropy")
    print("Evaluation: Gaussian phase regions (threshold 0.5)")

    train_loader, val_loader = build_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        split_path=args.split_path,
        earthquake_path=args.earthquake_path,
        noise_path=args.noise_path
    )

    model = PhaseNet(
        in_channels=3,
        num_classes=NUM_CLASSES
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4
    )

    best_val_loss = float("inf")
    history = []

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    history_path = (
        PROJECT_ROOT
        / "experiments"
        / "training_history.json"
    )

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nTraining and validation loaders ready.")
    print(f"Training batches: {len(train_loader):,}")
    print(f"Validation batches: {len(val_loader):,}")

    print("\nFinal test dataset: NOT LOADED")

    for epoch in range(1, args.epochs + 1):

        start_time = time.time()

        train_result = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            max_batches=args.max_train_batches
        )

        val_result = run_epoch(
            model,
            val_loader,
            device,
            max_batches=args.max_val_batches
        )

        elapsed = time.time() - start_time

        record = {
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "epoch": epoch,
            "train_loss": round(train_result["loss"], 6),
            "validation_loss": round(val_result["loss"], 6),
            "validation_metrics": val_result["metrics"],
            "validation_confusion_matrix": (
                val_result["confusion_matrix"]
            ),
            "train_batches": train_result["batches"],
            "validation_batches": val_result["batches"],
            "seconds": round(elapsed, 2),
            "loss_function": (
                "hybrid_standard_weighted_soft_cross_entropy"
            ),
            "phase_region_threshold": 0.5,
        }

        history.append(record)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Training loss:   {train_result['loss']:.6f}")
        print(f"Validation loss: {val_result['loss']:.6f}")

        print("\nValidation metrics:")

        for name, values in val_result["metrics"].items():
            print(
                f"{name:8s} | "
                f"IoU: {values['iou']:.4f} | "
                f"Precision: {values['precision']:.4f} | "
                f"Recall: {values['recall']:.4f} | "
                f"F1: {values['f1']:.4f}"
            )

        print(f"\nTime: {elapsed:.1f} seconds")

        if (
            not args.no_checkpoint
            and val_result["loss"] < best_val_loss
        ):

            best_val_loss = val_result["loss"]

            checkpoint_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": best_val_loss,
                    "validation_metrics": val_result["metrics"],
                    "class_names": CLASS_NAMES,
                    "seed": args.seed,
                    "loss_function": (
                        "hybrid_standard_weighted_soft_cross_entropy"
                    ),
                    "phase_region_threshold": 0.5,
                },
                checkpoint_path
            )

            print(f"Best checkpoint saved: {checkpoint_path}")

        # Save history after every epoch to avoid losing progress.
        with history_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=4)

    if not args.no_checkpoint:
        print(f"\nBest validation loss: {best_val_loss:.6f}")
        print(f"Checkpoint: {checkpoint_path}")

    print(f"\nTraining history saved: {history_path}")
    print("\nTRAINING RUN COMPLETED")


if __name__ == "__main__":
    main()
