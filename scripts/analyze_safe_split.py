
"""Analyze leakage-safe earthquake partitions and noise metadata."""

from collections import Counter
from pathlib import Path
import re

import h5py
import numpy as np


TRAIN_PATH = Path("data/raw/stead/train.hdf5")
TEST_PATH = Path("data/raw/stead/test.hdf5")
NOISE_PATH = Path("data/raw/stead/train_noise.hdf5")

EARTHQUAKE_PATTERN = re.compile(r"^(.+?)_(\d{14})_EV$")
NOISE_PATTERN = re.compile(r"^(.+?)_(\d{12,14})_NO$")


def read_metadata(path):
    values = []

    with h5py.File(path, "r") as hdf:
        metadata = hdf["metadata"]

        for start in range(0, len(metadata), 5000):
            batch = metadata[start:start + 5000]

            for value in batch:
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                values.append(str(value))

    return values


def earthquake_groups(metadata):
    groups = []
    unmatched = []

    for trace_id in metadata:
        match = EARTHQUAKE_PATTERN.match(trace_id)

        if match:
            station, timestamp = match.groups()
            groups.append(timestamp)
        else:
            unmatched.append(trace_id)

    return groups, unmatched


def main():
    for path in (TRAIN_PATH, TEST_PATH, NOISE_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    train_metadata = read_metadata(TRAIN_PATH)
    test_metadata = read_metadata(TEST_PATH)
    noise_metadata = read_metadata(NOISE_PATH)

    train_groups, train_unmatched = earthquake_groups(train_metadata)
    test_groups, test_unmatched = earthquake_groups(test_metadata)

    train_group_counts = Counter(train_groups)
    test_group_counts = Counter(test_groups)

    test_timestamps = set(test_group_counts)

    # Remove every training record whose timestamp occurs in test.
    excluded_indices = [
        index
        for index, timestamp in enumerate(train_groups)
        if timestamp in test_timestamps
    ]

    safe_indices = [
        index
        for index, timestamp in enumerate(train_groups)
        if timestamp not in test_timestamps
    ]

    safe_timestamps = {
        train_groups[index] for index in safe_indices
    }

    safe_group_sizes = Counter(
        train_groups[index] for index in safe_indices
    )

    print("\n========== EARTHQUAKE SPLIT ANALYSIS ==========")
    print(f"Original training records: {len(train_metadata)}")
    print(f"Original test records: {len(test_metadata)}")
    print(f"Shared candidate timestamps: {len(test_timestamps & set(train_groups))}")
    print(f"Training records excluded: {len(excluded_indices)}")
    print(f"Remaining safe training records: {len(safe_indices)}")
    print(f"Remaining candidate event groups: {len(safe_timestamps)}")
    print(f"Unmatched training IDs: {len(train_unmatched)}")
    print(f"Unmatched test IDs: {len(test_unmatched)}")

    print("\n========== SAFE GROUP DISTRIBUTION ==========")
    print(f"Largest group size: {max(safe_group_sizes.values())}")
    print(
        "Groups with multiple records:",
        sum(size > 1 for size in safe_group_sizes.values()),
    )

    print("\n========== NOISE METADATA ANALYSIS ==========")

    noise_timestamps = Counter()
    noise_stations = Counter()
    noise_unmatched = []

    for trace_id in noise_metadata:
        match = NOISE_PATTERN.match(trace_id)

        if match:
            station, timestamp = match.groups()
            noise_timestamps[timestamp] += 1
            noise_stations[station] += 1
        else:
            noise_unmatched.append(trace_id)

    print(f"Noise records: {len(noise_metadata)}")
    print(f"Unique noise timestamps: {len(noise_timestamps)}")
    print(
        "Repeated noise timestamps:",
        sum(count > 1 for count in noise_timestamps.values()),
    )
    print(f"Unique noise stations: {len(noise_stations)}")
    print(f"Unmatched noise IDs: {len(noise_unmatched)}")

    print("\nLargest noise timestamp groups:")
    for timestamp, count in noise_timestamps.most_common(10):
        print(f"{timestamp}: {count} records")

    print("\nFirst unmatched noise IDs:")
    for trace_id in noise_unmatched[:10]:
        print(trace_id)

    # Check that the proposed earthquake training pool is disjoint
    # from the existing earthquake test timestamps.
    assert safe_timestamps.isdisjoint(test_timestamps)

    print("\nSUCCESS: Candidate earthquake timestamp overlap removed.")
    print("No files were modified.")


if __name__ == "__main__":
    main()
