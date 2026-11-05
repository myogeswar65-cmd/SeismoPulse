
"""Create reproducible, leakage-aware SeismoPulse train/validation splits."""

from collections import defaultdict
from datetime import datetime, timedelta
from bisect import bisect_left, bisect_right
from pathlib import Path
import json
import random
import re

import h5py


EARTHQUAKE_TRAIN = Path("data/raw/stead/train.hdf5")
EARTHQUAKE_TEST = Path("data/raw/stead/test.hdf5")
NOISE_TRAIN = Path("data/raw/stead/train_noise.hdf5")
NOISE_TEST = Path("data/raw/stead/test_noise.hdf5")

OUTPUT_PATH = Path("data/metadata/dataset_split.json")

SEED = 42
VALIDATION_FRACTION = 0.20
WINDOW_SECONDS = 60

EARTHQUAKE_PATTERN = re.compile(r"^(.+?)_(\d{14})_EV$")
NOISE_PATTERN = re.compile(r"^(.+?)_(\d{12}|\d{14})_NO$")


def read_records(path, pattern, is_noise=False):
    records = []

    with h5py.File(path, "r") as hdf:
        for index, value in enumerate(hdf["metadata"]):
            if isinstance(value, bytes):
                value = value.decode("utf-8")

            trace_id = str(value)
            match = pattern.match(trace_id)

            if not match:
                raise ValueError(f"Invalid trace ID: {trace_id}")

            station, timestamp = match.groups()

            if is_noise and len(timestamp) == 12:
                start = datetime.strptime(timestamp, "%Y%m%d%H%M")
                end = start + timedelta(seconds=119)
            else:
                fmt = "%Y%m%d%H%M%S"
                start = datetime.strptime(timestamp, fmt)
                end = start + timedelta(seconds=WINDOW_SECONDS)

            records.append({
                "index": index,
                "id": trace_id,
                "station": station,
                "timestamp": timestamp,
                "start": start,
                "end": end,
            })

    return records


def has_nearby_time(times, start):
    """Check for another recording start strictly within 60 seconds."""
    lower = start - timedelta(seconds=WINDOW_SECONDS)
    upper = start + timedelta(seconds=WINDOW_SECONDS)

    position = bisect_right(times, lower)

    return position < len(times) and times[position] < upper


def filter_earthquake_test_overlap(train, test):
    test_timestamps = {record["timestamp"] for record in test}
    test_by_station = defaultdict(list)

    for record in test:
        test_by_station[record["station"]].append(record["start"])

    for station in test_by_station:
        test_by_station[station].sort()

    safe = []

    for record in train:
        if record["timestamp"] in test_timestamps:
            continue

        times = test_by_station.get(record["station"], [])

        if has_nearby_time(times, record["start"]):
            continue

        safe.append(record)

    return safe


def filter_noise_test_overlap(train, test):
    """Conservatively exclude noise windows overlapping any test window."""
    test_by_station = defaultdict(list)

    for record in test:
        test_by_station[record["station"]].append(
            (record["start"], record["end"])
        )

    # Merge test intervals for efficient overlap checks.
    merged_by_station = {}

    for station, intervals in test_by_station.items():
        intervals.sort()
        merged = []

        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)

        merged_by_station[station] = merged

    safe = []

    for record in train:
        intervals = merged_by_station.get(record["station"], [])

        if not intervals:
            safe.append(record)
            continue

        starts = [interval[0] for interval in intervals]
        position = bisect_right(starts, record["end"]) - 1

        overlaps_test = (
            position >= 0
            and intervals[position][1] >= record["start"]
        )

        if not overlaps_test:
            safe.append(record)

    return safe


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)

        if a == b:
            return

        if self.rank[a] < self.rank[b]:
            a, b = b, a

        self.parent[b] = a

        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def build_earthquake_components(records):
    """Connect shared candidate timestamps and same-station overlaps."""
    uf = UnionFind(len(records))

    timestamp_groups = defaultdict(list)
    station_groups = defaultdict(list)

    for i, record in enumerate(records):
        timestamp_groups[record["timestamp"]].append(i)
        station_groups[record["station"]].append(
            (record["start"], i)
        )

    for indices in timestamp_groups.values():
        first = indices[0]

        for other in indices[1:]:
            uf.union(first, other)

    for items in station_groups.values():
        items.sort()

        for i in range(len(items)):
            start_a, index_a = items[i]

            j = i + 1

            while j < len(items):
                start_b, index_b = items[j]

                gap = (start_b - start_a).total_seconds()

                if gap >= WINDOW_SECONDS:
                    break

                uf.union(index_a, index_b)
                j += 1

    components = defaultdict(list)

    for i, record in enumerate(records):
        components[uf.find(i)].append(record["index"])

    return list(components.values())


def split_components(components):
    """Assign complete earthquake components to an 80/20 split."""
    rng = random.Random(SEED)
    groups = list(components)
    rng.shuffle(groups)

    # Largest-first assignment reduces the final size imbalance.
    groups.sort(key=len, reverse=True)

    total_records = sum(len(group) for group in groups)
    target_validation = total_records * VALIDATION_FRACTION

    validation_groups = []
    training_groups = []
    validation_count = 0

    for group in groups:
        current_error = abs(validation_count - target_validation)
        added_error = abs(
            validation_count + len(group) - target_validation
        )

        if added_error < current_error:
            validation_groups.append(group)
            validation_count += len(group)
        else:
            training_groups.append(group)

    train_indices = sorted(
        index for group in training_groups for index in group
    )
    validation_indices = sorted(
        index for group in validation_groups for index in group
    )

    if not train_indices or not validation_indices:
        raise RuntimeError("Earthquake split produced an empty partition.")

    return train_indices, validation_indices, len(components)


def split_noise_chronologically(records):
    """
    Find a global time gap that separates complete recording windows.

    Every record before the selected gap goes to training; every record
    after it goes to validation. This prevents overlapping time windows
    across the two partitions, including recordings from different stations.
    """
    ordered = sorted(records, key=lambda record: record["start"])

    if len(ordered) < 2:
        raise RuntimeError("Not enough noise records to split.")

    target_train = len(ordered) * (1 - VALIDATION_FRACTION)

    prefix_max_end = ordered[0]["end"]
    best = None

    for position in range(1, len(ordered)):
        current = ordered[position]

        # The entire preceding interval set must end before this start.
        gap_seconds = (
            current["start"] - prefix_max_end
        ).total_seconds()

        if gap_seconds > 0:
            train_count = position
            error = abs(train_count - target_train)

            if best is None or error < best["error"]:
                best = {
                    "position": position,
                    "error": error,
                    "gap_seconds": gap_seconds,
                    "left_end": prefix_max_end,
                    "right_start": current["start"],
                }

        prefix_max_end = max(prefix_max_end, current["end"])

    if best is None:
        raise RuntimeError(
            "No global non-overlapping time boundary found for noise data."
        )

    training_records = ordered[:best["position"]]
    validation_records = ordered[best["position"]:]

    train_indices = sorted(record["index"] for record in training_records)
    validation_indices = sorted(
        record["index"] for record in validation_records
    )

    if not train_indices or not validation_indices:
        raise RuntimeError("Noise split produced an empty partition.")

    # Final global interval leakage check.
    latest_train_end = max(
        record["end"] for record in training_records
    )
    earliest_validation_start = min(
        record["start"] for record in validation_records
    )

    if latest_train_end >= earliest_validation_start:
        raise RuntimeError("Noise temporal leakage detected.")

    details = {
        "boundary_gap_seconds": best["gap_seconds"],
        "latest_training_window_end": latest_train_end.isoformat(),
        "earliest_validation_window_start": (
            earliest_validation_start.isoformat()
        ),
    }

    return train_indices, validation_indices, details


def verify_indices(train, validation, safe_records, name):
    train_set = set(train)
    validation_set = set(validation)
    safe_set = {record["index"] for record in safe_records}

    if train_set & validation_set:
        raise RuntimeError(f"{name}: train/validation index overlap.")

    if train_set | validation_set != safe_set:
        raise RuntimeError(f"{name}: records missing or unexpectedly added.")

    print(
        f"{name}: {len(train)} training / "
        f"{len(validation)} validation records"
    )


def main():
    print("Loading metadata...")

    earthquake_train = read_records(
        EARTHQUAKE_TRAIN, EARTHQUAKE_PATTERN
    )
    earthquake_test = read_records(
        EARTHQUAKE_TEST, EARTHQUAKE_PATTERN
    )

    noise_train = read_records(
        NOISE_TRAIN, NOISE_PATTERN, is_noise=True
    )
    noise_test = read_records(
        NOISE_TEST, NOISE_PATTERN, is_noise=True
    )

    print("Applying final-test leakage exclusions...")

    safe_earthquake = filter_earthquake_test_overlap(
        earthquake_train, earthquake_test
    )
    safe_noise = filter_noise_test_overlap(
        noise_train, noise_test
    )

    earthquake_excluded = (
        len(earthquake_train) - len(safe_earthquake)
    )
    noise_excluded = len(noise_train) - len(safe_noise)

    print(f"Earthquake exclusions: {earthquake_excluded}")
    print(f"Noise exclusions: {noise_excluded}")

    print("Building earthquake components...")

    components = build_earthquake_components(safe_earthquake)

    (
        earthquake_train_indices,
        earthquake_validation_indices,
        component_count,
    ) = split_components(components)

    print("Creating chronological noise split...")

    (
        noise_train_indices,
        noise_validation_indices,
        noise_boundary,
    ) = split_noise_chronologically(safe_noise)

    print("\n========== SPLIT VERIFICATION ==========")

    verify_indices(
        earthquake_train_indices,
        earthquake_validation_indices,
        safe_earthquake,
        "Earthquake",
    )

    verify_indices(
        noise_train_indices,
        noise_validation_indices,
        safe_noise,
        "Noise",
    )

    output = {
        "project": "SeismoPulse",
        "seed": SEED,
        "validation_fraction_target": VALIDATION_FRACTION,
        "dataset_sources": {
            "earthquake_train": str(EARTHQUAKE_TRAIN),
            "earthquake_test": str(EARTHQUAKE_TEST),
            "noise_train": str(NOISE_TRAIN),
            "noise_test": str(NOISE_TEST),
        },
        "exclusions": {
            "earthquake_training_records_excluded": earthquake_excluded,
            "noise_training_records_excluded": noise_excluded,
        },
        "earthquake": {
            "strategy": "80/20 connected-component split",
            "component_count": component_count,
            "train_indices": earthquake_train_indices,
            "validation_indices": earthquake_validation_indices,
            "final_test_count": len(earthquake_test),
        },
        "noise": {
            "strategy": "chronological split at a global non-overlapping gap",
            "boundary": noise_boundary,
            "train_indices": noise_train_indices,
            "validation_indices": noise_validation_indices,
            "final_test_count": len(noise_test),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print("\n========== FINAL COUNTS ==========")
    print(f"Earthquake safe records: {len(safe_earthquake)}")
    print(f"Noise safe records: {len(safe_noise)}")
    print(f"Earthquake components: {component_count}")
    print(
        "Noise boundary gap: "
        f"{noise_boundary['boundary_gap_seconds']:.2f} seconds"
    )
    print(f"Saved split: {OUTPUT_PATH}")
    print("\nSUCCESS: Split created and index checks passed.")
    print("Original HDF5 files were not modified.")


if __name__ == "__main__":
    main()
