
"""Inspect candidate timestamp groups and cross-file overlap."""

from collections import Counter
from pathlib import Path
import re

import h5py


FILES = {
    "earthquake_train": Path("data/raw/stead/train.hdf5"),
    "earthquake_test": Path("data/raw/stead/test.hdf5"),
    "noise_train": Path("data/raw/stead/train_noise.hdf5"),
    "noise_test": Path("data/raw/stead/test_noise.hdf5"),
}

PATTERN = re.compile(r"^(.+?)_(\d{14})_(EV|NO)$")


def inspect_file(path):
    timestamps = Counter()
    unmatched = []
    total = 0

    with h5py.File(path, "r") as hdf:
        metadata = hdf["metadata"]

        for start in range(0, len(metadata), 5000):
            for value in metadata[start:start + 5000]:
                if isinstance(value, bytes):
                    value = value.decode("utf-8")

                trace_id = str(value)
                total += 1

                match = PATTERN.match(trace_id)

                if match:
                    station, timestamp, category = match.groups()
                    timestamps[timestamp] += 1
                else:
                    unmatched.append(trace_id)

    return total, timestamps, unmatched


def main():
    results = {}

    print("========== FILE INSPECTION ==========")

    for name, path in FILES.items():
        if not path.is_file():
            print(f"\n{name}: FILE NOT FOUND")
            continue

        total, timestamps, unmatched = inspect_file(path)

        results[name] = timestamps

        repeated = sum(1 for count in timestamps.values() if count > 1)

        print(f"\n{name}")
        print(f"Records: {total}")
        print(f"Unique timestamps: {len(timestamps)}")
        print(f"Repeated timestamps: {repeated}")
        print(f"Unmatched IDs: {len(unmatched)}")

        if unmatched:
            print("Examples:", unmatched[:5])

    print("\n========== CROSS-FILE TIMESTAMP OVERLAP ==========")

    comparisons = [
        ("earthquake_train", "earthquake_test"),
        ("noise_train", "noise_test"),
        ("earthquake_train", "noise_train"),
        ("earthquake_train", "noise_test"),
    ]

    for first, second in comparisons:
        if first not in results or second not in results:
            continue

        overlap = set(results[first]) & set(results[second])

        print(f"\n{first} vs {second}")
        print(f"Shared candidate timestamps: {len(overlap)}")

        if overlap:
            print("Examples:", sorted(overlap)[:10])

    print("\nInspection complete.")
    print("No waveform data was loaded or modified.")


if __name__ == "__main__":
    main()
