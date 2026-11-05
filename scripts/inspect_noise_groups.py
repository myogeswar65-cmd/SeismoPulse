
"""Detailed inspection of noise-record metadata grouping."""

from collections import Counter, defaultdict
from pathlib import Path
import re

import h5py


NOISE_PATH = Path("data/raw/stead/train_noise.hdf5")

PATTERN = re.compile(r"^(.+?)_(\d{12,14})_NO$")


def main():
    timestamp_groups = defaultdict(list)
    station_counts = Counter()
    exact_ids = Counter()
    timestamp_lengths = Counter()
    unmatched = []

    with h5py.File(NOISE_PATH, "r") as hdf:
        metadata = hdf["metadata"]

        for start in range(0, len(metadata), 5000):
            for value in metadata[start:start + 5000]:

                if isinstance(value, bytes):
                    value = value.decode("utf-8")

                trace_id = str(value)
                exact_ids[trace_id] += 1

                match = PATTERN.match(trace_id)

                if not match:
                    unmatched.append(trace_id)
                    continue

                station, timestamp = match.groups()

                timestamp_groups[timestamp].append(station)
                station_counts[station] += 1
                timestamp_lengths[len(timestamp)] += 1

    duplicate_ids = {
        trace_id: count
        for trace_id, count in exact_ids.items()
        if count > 1
    }

    print("\n========== NOISE GROUP INVESTIGATION ==========")

    print(f"Total records: {sum(exact_ids.values())}")
    print(f"Unique exact IDs: {len(exact_ids)}")
    print(f"Duplicate exact IDs: {len(duplicate_ids)}")
    print(f"Unique timestamps: {len(timestamp_groups)}")
    print(f"Unique stations: {len(station_counts)}")
    print(f"Timestamp lengths: {dict(timestamp_lengths)}")
    print(f"Unmatched IDs: {len(unmatched)}")

    print("\n========== LARGEST TIMESTAMP GROUPS ==========")

    largest = sorted(
        timestamp_groups.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )[:15]

    for timestamp, stations in largest:
        print(f"\nTimestamp: {timestamp}")
        print(f"Records: {len(stations)}")
        print(f"Unique stations: {len(set(stations))}")
        print(f"Stations: {sorted(set(stations))[:20]}")

    print("\n========== DUPLICATE STATION-TIMESTAMP PAIRS ==========")

    duplicate_pairs = []

    for timestamp, stations in timestamp_groups.items():
        counts = Counter(stations)

        for station, count in counts.items():
            if count > 1:
                duplicate_pairs.append((station, timestamp, count))

    print(f"Duplicate station-timestamp pairs: {len(duplicate_pairs)}")

    for station, timestamp, count in duplicate_pairs[:15]:
        print(f"{station} | {timestamp} | {count} records")

    print("\n========== MOST FREQUENT STATIONS ==========")

    for station, count in station_counts.most_common(15):
        print(f"{station}: {count} records")

    print("\n========== EXAMPLE NOISE IDS ==========")

    for trace_id in list(exact_ids)[:20]:
        print(trace_id)

    if unmatched:
        print("\nUnmatched examples:")
        for trace_id in unmatched[:10]:
            print(trace_id)

    print("\nNoise metadata inspection complete.")
    print("No waveform data was loaded or modified.")


if __name__ == "__main__":
    main()
