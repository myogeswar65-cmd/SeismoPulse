"""Inspect noise training groups for leakage-aware validation splitting."""

from collections import Counter, defaultdict
from bisect import bisect_left
from datetime import datetime, timedelta
from pathlib import Path
import re

import h5py


TRAIN_PATH = Path("data/raw/stead/train_noise.hdf5")
TEST_PATH = Path("data/raw/stead/test_noise.hdf5")

PATTERN = re.compile(r"^(.+?)_(\d{12}|\d{14})_NO$")


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)

        if a == b:
            return

        if self.rank[a] < self.rank[b]:
            a, b = b, a

        self.parent[b] = a

        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def read_records(path):
    records = []

    with h5py.File(path, "r") as hdf:
        for index, value in enumerate(hdf["metadata"]):
            if isinstance(value, bytes):
                value = value.decode("utf-8")

            trace_id = str(value)
            match = PATTERN.match(trace_id)

            if not match:
                raise ValueError(f"Invalid noise trace ID: {trace_id}")

            station, timestamp = match.groups()

            if len(timestamp) == 14:
                start = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
                end = start + timedelta(seconds=60)
                group_key = timestamp
            else:
                start = datetime.strptime(timestamp, "%Y%m%d%H%M")
                end = start + timedelta(seconds=119)
                group_key = timestamp

            records.append({
                "index": index,
                "id": trace_id,
                "station": station,
                "timestamp": timestamp,
                "group_key": group_key,
                "start": start,
                "end": end,
            })

    return records


def overlaps(a, b):
    return a["start"] <= b["end"] and b["start"] <= a["end"]


def main():
    train = read_records(TRAIN_PATH)
    test = read_records(TEST_PATH)

    # Build test intervals by station.
    test_by_station = defaultdict(list)

    for record in test:
        test_by_station[record["station"]].append(record)

    for station in test_by_station:
        test_by_station[station].sort(
            key=lambda record: record["start"]
        )

    # Exclude every training record that may overlap a test window.
    safe_records = []

    for record in train:
        test_intervals = test_by_station.get(record["station"], [])

        starts = [item["start"] for item in test_intervals]
        position = bisect_left(starts, record["end"])

        # Check the nearest possible earlier interval.
        possible = test_intervals[:position + 1]

        has_overlap = any(
            overlaps(record, test_record)
            for test_record in possible
        )

        if not has_overlap:
            safe_records.append(record)

    # Connect records sharing a candidate timestamp.
    uf = UnionFind(len(safe_records))
    timestamp_groups = defaultdict(list)

    for i, record in enumerate(safe_records):
        timestamp_groups[record["group_key"]].append(i)

    for indices in timestamp_groups.values():
        first = indices[0]

        for other in indices[1:]:
            uf.union(first, other)

    # Connect overlapping windows from the same station.
    station_groups = defaultdict(list)

    for i, record in enumerate(safe_records):
        station_groups[record["station"]].append((record, i))

    for station, items in station_groups.items():
        items.sort(key=lambda item: item[0]["start"])

        for i in range(len(items)):
            record_a, index_a = items[i]

            j = i + 1

            while j < len(items):
                record_b, index_b = items[j]

                if record_b["start"] > record_a["end"]:
                    break

                uf.union(index_a, index_b)
                j += 1

    # Collect connected components.
    components = defaultdict(list)

    for i, record in enumerate(safe_records):
        components[uf.find(i)].append(record)

    sizes = Counter(len(group) for group in components.values())

    largest = sorted(
        components.values(),
        key=len,
        reverse=True,
    )[:15]

    print("\n========== NOISE TRAINING GROUP ANALYSIS ==========")
    print(f"Original training records: {len(train)}")
    print(f"Final test records: {len(test)}")
    print(f"Safe noise records: {len(safe_records)}")

    print("\n========== CONNECTED COMPONENTS ==========")
    print(f"Total components: {len(components)}")
    print(f"Singleton components: {sizes.get(1, 0)}")
    print(
        "Components with multiple records: "
        f"{sum(count for size, count in sizes.items() if size > 1)}"
    )
    print(f"Largest component: {max(sizes)} records")

    print("\n========== COMPONENT SIZE DISTRIBUTION ==========")

    for size, count in sorted(sizes.items()):
        if size <= 10:
            print(f"{size} records: {count} components")

    print("\n========== LARGEST COMPONENTS ==========")

    for number, group in enumerate(largest, start=1):
        stations = {record["station"] for record in group}
        timestamps = {record["group_key"] for record in group}

        print(f"\nComponent {number}")
        print(f"Records: {len(group)}")
        print(f"Stations: {len(stations)}")
        print(f"Candidate timestamps: {len(timestamps)}")
        print(f"Examples: {[r['id'] for r in group[:5]]}")

    print("\nAnalysis complete.")
    print("No waveform data was loaded or modified.")
    print("No dataset files were changed.")


if __name__ == "__main__":
    main()