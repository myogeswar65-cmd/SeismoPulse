"""Inspect the global time distribution of safe noise training records."""

from collections import Counter
from bisect import bisect_left
from datetime import datetime, timedelta
from pathlib import Path
import re

import h5py


TRAIN_PATH = Path("data/raw/stead/train_noise.hdf5")
TEST_PATH = Path("data/raw/stead/test_noise.hdf5")

PATTERN = re.compile(r"^(.+?)_(\d{12}|\d{14})_NO$")


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
                start = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
                end = start + timedelta(seconds=60)
            else:
                start = datetime.strptime(timestamp, "%Y%m%d%H%M")
                end = start + timedelta(seconds=119)

            records.append({
                "index": index,
                "id": trace_id,
                "station": station,
                "timestamp": timestamp,
                "start": start,
                "end": end,
            })

    return records


def overlaps(a, b):
    return a["start"] <= b["end"] and b["start"] <= a["end"]


def main():
    train = read_records(TRAIN_PATH)
    test = read_records(TEST_PATH)

    test_by_station = {}

    for record in test:
        test_by_station.setdefault(record["station"], []).append(record)

    for station in test_by_station:
        test_by_station[station].sort(
            key=lambda item: item["start"]
        )

    safe_records = []

    for record in train:
        intervals = test_by_station.get(record["station"], [])
        starts = [item["start"] for item in intervals]

        position = bisect_left(starts, record["end"])

        possible = intervals[:position + 1]

        if not any(overlaps(record, item) for item in possible):
            safe_records.append(record)

    safe_records.sort(key=lambda item: item["start"])

    daily = Counter(
        record["start"].strftime("%Y-%m-%d")
        for record in safe_records
    )

    hourly = Counter(
        record["start"].strftime("%Y-%m-%d %H")
        for record in safe_records
    )

    timestamp_lengths = Counter(
        len(record["timestamp"])
        for record in safe_records
    )

    unique_stations = {
        record["station"] for record in safe_records
    }

    unique_timestamps = {
        record["timestamp"] for record in safe_records
    }

    print("\n========== NOISE TIME DISTRIBUTION ==========")
    print(f"Safe records: {len(safe_records)}")
    print(f"Unique stations: {len(unique_stations)}")
    print(f"Unique candidate timestamps: {len(unique_timestamps)}")
    print(f"Timestamp lengths: {dict(timestamp_lengths)}")

    print("\n========== OVERALL TIME RANGE ==========")

    if safe_records:
        print(f"Earliest recording: {safe_records[0]['start']}")
        print(f"Latest recording: {safe_records[-1]['start']}")
        print(
            "Total time span: "
            f"{safe_records[-1]['start'] - safe_records[0]['start']}"
        )

    print("\n========== RECORDS PER DATE ==========")

    for date, count in sorted(daily.items()):
        print(f"{date}: {count}")

    print("\n========== RECORDS PER HOUR ==========")

    for hour, count in sorted(hourly.items()):
        print(f"{hour}: {count}")

    print("\n========== GLOBAL RECORDING GAPS ==========")

    gaps = Counter()

    for previous, current in zip(
        safe_records,
        safe_records[1:],
    ):
        seconds = (current["start"] - previous["start"]).total_seconds()

        if seconds < 60:
            gaps["under_60_seconds"] += 1
        elif seconds == 60:
            gaps["exactly_60_seconds"] += 1
        else:
            gaps["over_60_seconds"] += 1

    print(dict(gaps))

    print("\n========== LARGEST DAILY GROUPS ==========")

    for date, count in daily.most_common(10):
        print(f"{date}: {count}")

    print("\nAnalysis complete.")
    print("No dataset files were modified.")


if __name__ == "__main__":
    main()