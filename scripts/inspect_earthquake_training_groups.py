"""Inspect connected earthquake groups for safe train/validation splitting."""

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import re

import h5py


TRAIN_PATH = Path("data/raw/stead/train.hdf5")
TEST_PATH = Path("data/raw/stead/test.hdf5")

PATTERN = re.compile(r"^(.+?)_(\d{14})_EV$")

WINDOW_SECONDS = 60


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


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

            records.append({
                "index": index,
                "id": trace_id,
                "station": station,
                "timestamp": timestamp,
                "start": datetime.strptime(
                    timestamp, "%Y%m%d%H%M%S"
                ),
            })

    return records


def main():
    train = read_records(TRAIN_PATH)
    test = read_records(TEST_PATH)

    test_timestamps = {
        record["timestamp"] for record in test
    }

    test_by_station = defaultdict(list)

    for record in test:
        test_by_station[record["station"]].append(
            record["start"]
        )

    for station in test_by_station:
        test_by_station[station].sort()

    # Exclude shared timestamps and same-station test overlaps.
    safe_records = []

    for record in train:
        if record["timestamp"] in test_timestamps:
            continue

        station = record["station"]
        test_times = test_by_station.get(station, [])
        start = record["start"]

        overlaps_test = any(
            abs((start - test_start).total_seconds())
            < WINDOW_SECONDS
            for test_start in test_times
        )

        if not overlaps_test:
            safe_records.append(record)

    uf = UnionFind(len(safe_records))

    # Relationship 1: same candidate timestamp.
    timestamp_groups = defaultdict(list)

    for i, record in enumerate(safe_records):
        timestamp_groups[record["timestamp"]].append(i)

    for indices in timestamp_groups.values():
        first = indices[0]

        for other in indices[1:]:
            uf.union(first, other)

    # Relationship 2: overlapping windows at the same station.
    station_groups = defaultdict(list)

    for i, record in enumerate(safe_records):
        station_groups[record["station"]].append(
            (record["start"], i)
        )

    for station, items in station_groups.items():
        items.sort()

        for i in range(len(items)):
            start_a, index_a = items[i]

            j = i + 1

            while j < len(items):
                start_b, index_b = items[j]

                gap = (start_b - start_a).total_seconds()

                if gap >= WINDOW_SECONDS:
                    break

                uf.union(index_a, index_b)
                j += 1

    components = defaultdict(list)

    for i, record in enumerate(safe_records):
        components[uf.find(i)].append(record)

    sizes = Counter(
        len(records) for records in components.values()
    )

    largest = sorted(
        components.values(),
        key=len,
        reverse=True,
    )[:15]

    print("\n========== EARTHQUAKE TRAINING GROUP ANALYSIS ==========")
    print(f"Original training records: {len(train)}")
    print(f"Final test records: {len(test)}")
    print(f"Safe earthquake records: {len(safe_records)}")

    print("\n========== CONNECTED COMPONENTS ==========")
    print(f"Total components: {len(components)}")
    print(f"Singleton components: {sizes.get(1, 0)}")
    print(f"Components with multiple records: {sum(v for k, v in sizes.items() if k > 1)}")
    print(f"Largest component: {max(sizes)} records")

    print("\n========== COMPONENT SIZE DISTRIBUTION ==========")

    for size, count in sorted(sizes.items()):
        if size <= 10:
            print(f"{size} records: {count} components")

    print("\n========== LARGEST COMPONENTS ==========")

    for number, group in enumerate(largest, start=1):
        stations = {r["station"] for r in group}
        timestamps = {r["timestamp"] for r in group}

        print(f"\nComponent {number}")
        print(f"Records: {len(group)}")
        print(f"Stations: {len(stations)}")
        print(f"Candidate timestamps: {len(timestamps)}")
        print(f"Examples: {[r['id'] for r in group[:5]]}")

    print("\nAnalysis complete.")
    print("No dataset files were modified.")
    print("No split files were created.")


if __name__ == "__main__":
    main()