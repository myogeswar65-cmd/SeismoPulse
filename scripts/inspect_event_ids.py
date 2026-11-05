
"""Inspect earthquake metadata for possible event-level grouping."""

from collections import Counter, defaultdict
from pathlib import Path
import re

import h5py


DATASET_PATH = Path("data/raw/stead/train.hdf5")
PATTERN = re.compile(r"^(.+?)_(\d{14})_(EV|NO)$")


def decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def main():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(DATASET_PATH)

    timestamp_groups = defaultdict(list)
    station_counts = Counter()
    category_counts = Counter()
    exact_ids = Counter()
    unmatched = []

    with h5py.File(DATASET_PATH, "r") as hdf:
        metadata = hdf["metadata"]
        total = len(metadata)

        print(f"Total earthquake records: {total}")

        # Read metadata in chunks to avoid unnecessary memory use.
        chunk_size = 5000

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)

            for value in metadata[start:end]:
                trace_id = decode(value)
                exact_ids[trace_id] += 1

                match = PATTERN.match(trace_id)

                if not match:
                    unmatched.append(trace_id)
                    continue

                station, timestamp, category = match.groups()

                timestamp_groups[timestamp].append(station)
                station_counts[station] += 1
                category_counts[category] += 1

    repeated_ids = {
        trace_id: count
        for trace_id, count in exact_ids.items()
        if count > 1
    }

    repeated_timestamps = {
        timestamp: stations
        for timestamp, stations in timestamp_groups.items()
        if len(stations) > 1
    }

    multi_station_groups = {
        timestamp: stations
        for timestamp, stations in repeated_timestamps.items()
        if len(set(stations)) > 1
    }

    print("\n========== METADATA INSPECTION ==========")
    print(f"Unique trace IDs: {len(exact_ids)}")
    print(f"Duplicate exact trace IDs: {len(repeated_ids)}")
    print(f"Unique candidate timestamps: {len(timestamp_groups)}")
    print(f"Repeated candidate timestamps: {len(repeated_timestamps)}")
    print(f"Timestamps shared across stations: {len(multi_station_groups)}")
    print(f"Unique stations: {len(station_counts)}")
    print(f"Unmatched metadata IDs: {len(unmatched)}")
    print(f"Categories: {dict(category_counts)}")

    print("\n========== EXAMPLE METADATA ==========")
    for trace_id in list(exact_ids)[:10]:
        print(trace_id)

    print("\n========== LARGEST MULTI-STATION GROUPS ==========")

    largest_groups = sorted(
        multi_station_groups.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )[:10]

    for timestamp, stations in largest_groups:
        print(
            f"{timestamp}: {len(stations)} records, "
            f"{len(set(stations))} unique stations"
        )
        print(f"  Stations: {sorted(set(stations))[:10]}")

    if unmatched:
        print("\nFirst unmatched IDs:")
        for trace_id in unmatched[:10]:
            print(trace_id)

    print("\nInspection complete.")
    print(
        "IMPORTANT: A shared timestamp is only a candidate grouping key. "
        "Do not use it for data splitting until its event-level meaning "
        "has been verified."
    )


if __name__ == "__main__":
    main()
