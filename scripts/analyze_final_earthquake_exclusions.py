"""Combine timestamp and same-station overlap exclusions."""

from collections import defaultdict
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from pathlib import Path
import re

import h5py


TRAIN_PATH = Path("data/raw/stead/train.hdf5")
TEST_PATH = Path("data/raw/stead/test.hdf5")

PATTERN = re.compile(r"^(.+?)_(\d{14})_EV$")

WINDOW_SECONDS = 60


def read_records(path):
    records = []

    with h5py.File(path, "r") as hdf:
        for index, value in enumerate(hdf["metadata"]):
            if isinstance(value, bytes):
                value = value.decode("utf-8")

            trace_id = str(value)
            match = PATTERN.match(trace_id)

            if not match:
                raise ValueError(f"Unmatched trace ID: {trace_id}")

            station, timestamp = match.groups()
            start = datetime.strptime(timestamp, "%Y%m%d%H%M%S")

            records.append({
                "index": index,
                "station": station,
                "timestamp": timestamp,
                "start": start,
                "trace_id": trace_id,
            })

    return records


def main():
    train = read_records(TRAIN_PATH)
    test = read_records(TEST_PATH)

    # Condition 1: candidate timestamps shared by train and test.
    test_timestamps = {
        record["timestamp"] for record in test
    }

    timestamp_exclusions = {
        record["index"]
        for record in train
        if record["timestamp"] in test_timestamps
    }

    # Condition 2: same-station waveform windows overlap.
    test_by_station = defaultdict(list)

    for record in test:
        test_by_station[record["station"]].append(
            record["start"]
        )

    for station in test_by_station:
        test_by_station[station].sort()

    overlap_exclusions = set()
    examples = []

    window = timedelta(seconds=WINDOW_SECONDS)

    for record in train:
        station = record["station"]
        start = record["start"]
        test_times = test_by_station.get(station, [])

        # Only inspect test starts strictly less than 60 seconds away.
        left = bisect_right(test_times, start - window)
        right = bisect_left(test_times, start + window)

        if left < right:
            overlap_exclusions.add(record["index"])

            if len(examples) < 10:
                examples.append((
                    record["trace_id"],
                    test_times[left],
                ))

    # Combine exclusions without double-counting.
    combined_exclusions = (
        timestamp_exclusions | overlap_exclusions
    )

    intersection = (
        timestamp_exclusions & overlap_exclusions
    )

    safe_indices = [
        record["index"]
        for record in train
        if record["index"] not in combined_exclusions
    ]

    safe_timestamps = {
        train[index]["timestamp"]
        for index in safe_indices
    }

    print("\n========== FINAL EARTHQUAKE EXCLUSION ANALYSIS ==========")

    print(f"Original training records: {len(train)}")
    print(f"Test records: {len(test)}")

    print("\n========== EXCLUSION CONDITIONS ==========")

    print(
        "Shared candidate timestamp exclusions: "
        f"{len(timestamp_exclusions)}"
    )

    print(
        "Same-station overlapping-window exclusions: "
        f"{len(overlap_exclusions)}"
    )

    print(
        "Records appearing in both exclusion sets: "
        f"{len(intersection)}"
    )

    print(
        "Combined unique exclusions: "
        f"{len(combined_exclusions)}"
    )

    print("\n========== FINAL SAFE TRAINING POOL ==========")

    print(f"Remaining training records: {len(safe_indices)}")
    print(f"Remaining candidate timestamp groups: {len(safe_timestamps)}")

    print("\n========== OVERLAP EXAMPLES ==========")

    for train_id, test_start in examples:
        print(f"Train: {train_id}")
        print(f"Test start: {test_start}")
        print()

    print("\n========== SAFETY CHECKS ==========")

    assert not (
        timestamp_exclusions - combined_exclusions
    )

    assert not (
        overlap_exclusions - combined_exclusions
    )

    assert len(safe_indices) + len(combined_exclusions) == len(train)

    print("All exclusion conditions included.")
    print("No records were deleted or modified.")
    print("No split files were created.")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()