"""Inspect time gaps between noise recordings at each station."""

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import re

import h5py


NOISE_PATH = Path("data/raw/stead/train_noise.hdf5")
PATTERN = re.compile(r"^(.+?)_(\d{12,14})_NO$")


def parse_timestamp(value):
    if len(value) == 12:
        return datetime.strptime(value, "%Y%m%d%H%M")

    return datetime.strptime(value, "%Y%m%d%H%M%S")


def main():
    station_records = defaultdict(list)
    unmatched = []

    with h5py.File(NOISE_PATH, "r") as hdf:
        metadata = hdf["metadata"]

        for value in metadata:
            if isinstance(value, bytes):
                value = value.decode("utf-8")

            trace_id = str(value)
            match = PATTERN.match(trace_id)

            if not match:
                unmatched.append(trace_id)
                continue

            station, timestamp = match.groups()

            station_records[station].append(
                (parse_timestamp(timestamp), trace_id)
            )

    gap_counts = {
        "under_60": 0,
        "exactly_60": 0,
        "over_60": 0,
    }

    short_gap_examples = []
    station_gap_counts = defaultdict(int)

    for station, records in station_records.items():
        records.sort(key=lambda item: item[0])

        for previous, current in zip(records, records[1:]):
            gap = (current[0] - previous[0]).total_seconds()

            if gap < 60:
                gap_counts["under_60"] += 1
                station_gap_counts[station] += 1

                if len(short_gap_examples) < 30:
                    short_gap_examples.append(
                        (station, gap, previous[1], current[1])
                    )

            elif gap == 60:
                gap_counts["exactly_60"] += 1

            else:
                gap_counts["over_60"] += 1

    print("\n========== NOISE TIME GAP ANALYSIS ==========")
    print(f"Stations inspected: {len(station_records)}")
    print(f"Unmatched IDs: {len(unmatched)}")

    print("\n========== CONSECUTIVE RECORD GAPS ==========")
    print(f"Gaps under 60 seconds: {gap_counts['under_60']}")
    print(f"Gaps exactly 60 seconds: {gap_counts['exactly_60']}")
    print(f"Gaps over 60 seconds: {gap_counts['over_60']}")

    print("\n========== STATIONS WITH MOST SHORT GAPS ==========")

    for station, count in sorted(
        station_gap_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:15]:
        print(f"{station}: {count}")

    print("\n========== EXAMPLES OF GAPS UNDER 60 SECONDS ==========")

    for station, gap, first, second in short_gap_examples:
        print(f"\nStation: {station}")
        print(f"Gap: {gap} seconds")
        print(f"First:  {first}")
        print(f"Second: {second}")

    print("\nTime gap inspection complete.")
    print("No waveform data was loaded or modified.")


if __name__ == "__main__":
    main()