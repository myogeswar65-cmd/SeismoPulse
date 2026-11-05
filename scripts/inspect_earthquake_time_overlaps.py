"""Check overlapping earthquake windows between train and test."""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import re

import h5py


TRAIN_PATH = Path("data/raw/stead/train.hdf5")
TEST_PATH = Path("data/raw/stead/test.hdf5")

PATTERN = re.compile(r"^(.+?)_(\d{12,14})_EV$")

WINDOW_SECONDS = 60


def parse_trace_id(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")

    trace_id = str(value)
    match = PATTERN.match(trace_id)

    if not match:
        return None

    station, timestamp = match.groups()

    # Minute-resolution IDs do not provide exact start seconds.
    if len(timestamp) != 14:
        return None

    start = datetime.strptime(timestamp, "%Y%m%d%H%M%S")

    return station, start, trace_id


def load_records(path):
    records = []
    unmatched = []
    minute_resolution = 0

    with h5py.File(path, "r") as hdf:
        metadata = hdf["metadata"]

        for index, value in enumerate(metadata):
            if isinstance(value, bytes):
                value = value.decode("utf-8")

            trace_id = str(value)
            match = PATTERN.match(trace_id)

            if not match:
                unmatched.append(trace_id)
                continue

            if len(match.group(2)) != 14:
                minute_resolution += 1
                continue

            parsed = parse_trace_id(trace_id)

            if parsed:
                station, start, _ = parsed
                records.append((index, station, start, trace_id))

    return records, unmatched, minute_resolution


def main():
    train, train_unmatched, train_minute = load_records(TRAIN_PATH)
    test, test_unmatched, test_minute = load_records(TEST_PATH)

    test_by_station = defaultdict(list)

    for index, station, start, trace_id in test:
        test_by_station[station].append(
            (start, index, trace_id)
        )

    for station in test_by_station:
        test_by_station[station].sort()

    overlapping_train = set()
    overlap_examples = []

    for train_index, station, train_start, train_id in train:
        for test_start, test_index, test_id in test_by_station.get(
            station, []
        ):
            gap = abs((train_start - test_start).total_seconds())

            if gap < WINDOW_SECONDS:
                overlapping_train.add(train_index)

                if len(overlap_examples) < 30:
                    overlap_examples.append(
                        (station, gap, train_id, test_id)
                    )

                break

    print("\n========== EARTHQUAKE TIME OVERLAP ANALYSIS ==========")
    print(f"Train records with exact timestamps: {len(train)}")
    print(f"Test records with exact timestamps: {len(test)}")

    print(f"Train IDs with minute-only timestamps: {train_minute}")
    print(f"Test IDs with minute-only timestamps: {test_minute}")

    print(f"Unmatched train IDs: {len(train_unmatched)}")
    print(f"Unmatched test IDs: {len(test_unmatched)}")

    print("\n========== SAME-STATION OVERLAPS ==========")
    print(
        "Training records overlapping a test window: "
        f"{len(overlapping_train)}"
    )

    print("\n========== EXAMPLES ==========")

    for station, gap, train_id, test_id in overlap_examples:
        print(f"\nStation: {station}")
        print(f"Start-time difference: {gap} seconds")
        print(f"Train: {train_id}")
        print(f"Test:  {test_id}")

    print("\nAnalysis complete.")
    print("No waveform data was loaded or modified.")


if __name__ == "__main__":
    main()