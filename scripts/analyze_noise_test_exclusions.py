"""Find noise training recordings that may overlap the independent test set."""

from collections import defaultdict
from bisect import bisect_left
from datetime import datetime, timedelta
from pathlib import Path
import re

import h5py


TRAIN_PATH = Path("data/raw/stead/train_noise.hdf5")
TEST_PATH = Path("data/raw/stead/test_noise.hdf5")

PATTERN = re.compile(r"^(.+?)_(\d{12,14})_NO$")

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
                raise ValueError(f"Invalid trace ID: {trace_id}")

            station, timestamp = match.groups()

            if len(timestamp) == 14:
                start = datetime.strptime(
                    timestamp, "%Y%m%d%H%M%S"
                )
                uncertainty = 0
            else:
                start = datetime.strptime(
                    timestamp, "%Y%m%d%H%M"
                )
                uncertainty = 59

            records.append({
                "index": index,
                "id": trace_id,
                "station": station,
                "start": start,
                "uncertainty": uncertainty,
            })

    return records


def main():
    train = read_records(TRAIN_PATH)
    test = read_records(TEST_PATH)

    # A 14-digit ID has an exact start time.
    # A 12-digit ID may start anywhere within its named minute.
    # We conservatively expand its possible recording interval.

    test_by_station = defaultdict(list)

    for record in test:
        start = record["start"]
        end = (
            start
            + timedelta(seconds=record["uncertainty"] + WINDOW_SECONDS)
        )

        test_by_station[record["station"]].append(
            (start, end, record["id"])
        )

    for station in test_by_station:
        test_by_station[station].sort(key=lambda item: item[0])

    test_starts = {
        station: [item[0] for item in intervals]
        for station, intervals in test_by_station.items()
    }

    excluded_train = set()
    examples = []

    for record in train:
        station = record["station"]

        if station not in test_by_station:
            continue

        train_start = record["start"]
        train_end = train_start + timedelta(
            seconds=record["uncertainty"] + WINDOW_SECONDS
        )

        intervals = test_by_station[station]
        starts = test_starts[station]

        # Test intervals starting before train_end may overlap.
        stop = bisect_left(starts, train_end)

        # A test interval can extend at most 119 seconds
        # beyond its recorded minute.
        earliest = train_start - timedelta(seconds=119)
        begin = bisect_left(starts, earliest)

        for test_start, test_end, test_id in intervals[begin:stop]:
            if train_start < test_end and test_start < train_end:
                excluded_train.add(record["index"])

                if len(examples) < 20:
                    examples.append((
                        record["id"],
                        test_id,
                    ))
                break

    stations_train = {r["station"] for r in train}
    stations_test = {r["station"] for r in test}

    print("\n========== NOISE TEST EXCLUSION ANALYSIS ==========")
    print(f"Noise training records: {len(train)}")
    print(f"Noise test records: {len(test)}")

    print("\n========== STATION COVERAGE ==========")
    print(f"Training stations: {len(stations_train)}")
    print(f"Test stations: {len(stations_test)}")
    print(f"Stations appearing in both: {len(stations_train & stations_test)}")

    print("\n========== TRAINING EXCLUSIONS ==========")
    print(f"Potentially overlapping training records: {len(excluded_train)}")
    print(f"Remaining noise training records: {len(train) - len(excluded_train)}")

    print("\n========== EXAMPLES ==========")

    for train_id, test_id in examples:
        print(f"Train: {train_id}")
        print(f"Test:  {test_id}")
        print()

    assert len(excluded_train) <= len(train)

    print("Analysis complete.")
    print("No waveform data was loaded or modified.")
    print("No dataset files were changed.")


if __name__ == "__main__":
    main()