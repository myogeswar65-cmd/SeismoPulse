"""Validate independent noise test data and check train/test overlap."""

from collections import Counter, defaultdict
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from pathlib import Path
import re

import h5py
import numpy as np


TRAIN_PATH = Path("data/raw/stead/train_noise.hdf5")
TEST_PATH = Path("data/raw/stead/test_noise.hdf5")

PATTERN = re.compile(r"^(.+?)_(\d{12,14})_NO$")


def parse_id(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")

    trace_id = str(value)
    match = PATTERN.match(trace_id)

    if not match:
        return None

    station, timestamp = match.groups()

    if len(timestamp) == 14:
        start = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
    else:
        start = datetime.strptime(timestamp, "%Y%m%d%H%M")

    return {
        "id": trace_id,
        "station": station,
        "timestamp": timestamp,
        "start": start,
        "precision": len(timestamp),
    }


def inspect_file(path):
    records = []
    unmatched = []
    timestamp_lengths = Counter()
    duplicate_ids = Counter()

    with h5py.File(path, "r") as hdf:
        print(f"\nFile: {path}")
        print(f"Root keys: {list(hdf.keys())}")

        if "metadata" not in hdf or "traces" not in hdf:
            raise ValueError("Expected metadata and traces datasets.")

        metadata = hdf["metadata"]
        traces = hdf["traces"]

        print(f"Metadata count: {len(metadata)}")
        print(f"Waveform shape: {traces.shape}")
        print(f"Waveform dtype: {traces.dtype}")

        if len(metadata) != len(traces):
            raise ValueError("Metadata and waveform counts do not match.")

        for value in metadata:
            parsed = parse_id(value)

            if parsed is None:
                unmatched.append(str(value))
                continue

            records.append(parsed)
            timestamp_lengths[parsed["precision"]] += 1
            duplicate_ids[parsed["id"]] += 1

        # Inspect a few waveforms without loading the entire file.
        sample_checks = []

        for index in range(min(5, len(traces))):
            waveform = traces[index]
            sample_checks.append({
                "index": index,
                "shape": waveform.shape,
                "finite": bool(np.isfinite(waveform).all()),
            })

    duplicates = {
        trace_id: count
        for trace_id, count in duplicate_ids.items()
        if count > 1
    }

    return records, unmatched, timestamp_lengths, duplicates, sample_checks


def main():
    train, train_unmatched, train_lengths, train_duplicates, _ = (
        inspect_file(TRAIN_PATH)
    )

    test, test_unmatched, test_lengths, test_duplicates, sample_checks = (
        inspect_file(TEST_PATH)
    )

    print("\n========== NOISE TEST VALIDATION ==========")
    print(f"Training records: {len(train)}")
    print(f"Test records: {len(test)}")
    print(f"Train timestamp lengths: {dict(train_lengths)}")
    print(f"Test timestamp lengths: {dict(test_lengths)}")
    print(f"Train unmatched IDs: {len(train_unmatched)}")
    print(f"Test unmatched IDs: {len(test_unmatched)}")
    print(f"Train duplicate IDs: {len(train_duplicates)}")
    print(f"Test duplicate IDs: {len(test_duplicates)}")

    print("\n========== SAMPLE WAVEFORM CHECKS ==========")

    for check in sample_checks:
        print(check)

    # Check exact ID overlap.
    train_ids = {record["id"] for record in train}
    test_ids = {record["id"] for record in test}

    exact_overlap = train_ids & test_ids

    print("\n========== TRAIN/TEST OVERLAP ==========")
    print(f"Exact duplicate IDs across files: {len(exact_overlap)}")

    # Check same-station temporal overlap for exact-second IDs.
    train_by_station = defaultdict(list)
    test_by_station = defaultdict(list)

    for record in train:
        if record["precision"] == 14:
            train_by_station[record["station"]].append(record["start"])

    for record in test:
        if record["precision"] == 14:
            test_by_station[record["station"]].append(record["start"])

    window = timedelta(seconds=60)
    overlapping_test_records = set()
    overlap_examples = []

    for station, test_times in test_by_station.items():
        train_times = sorted(train_by_station.get(station, []))

        for test_start in test_times:
            left = bisect_right(train_times, test_start - window)
            right = bisect_left(train_times, test_start + window)

            if left < right:
                overlapping_test_records.add((station, test_start))

                if len(overlap_examples) < 20:
                    overlap_examples.append((
                        station,
                        test_start,
                        train_times[left],
                    ))

    print(
        "Test records with same-station overlap against training: "
        f"{len(overlapping_test_records)}"
    )

    print("\n========== OVERLAP EXAMPLES ==========")

    for station, test_start, train_start in overlap_examples:
        gap = abs((test_start - train_start).total_seconds())
        print(f"{station}: train/test gap = {gap} seconds")

    print("\n========== FINAL CHECKS ==========")

    if train_unmatched or test_unmatched:
        print("WARNING: Some metadata IDs could not be parsed.")

    if exact_overlap:
        print("WARNING: Exact record IDs occur in both files.")

    if overlapping_test_records:
        print("WARNING: Same-station overlapping windows were found.")

    if not (
        train_unmatched
        or test_unmatched
        or exact_overlap
        or overlapping_test_records
        or train_duplicates
        or test_duplicates
    ):
        print("All inspected metadata and overlap checks passed.")

    print("\nNo dataset files were modified.")


if __name__ == "__main__":
    main()