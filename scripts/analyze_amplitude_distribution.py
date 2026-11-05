"""Investigate amplitude distribution in STEAD."""

import h5py
import numpy as np

FILE_PATH = "data/raw/stead/test.hdf5"


def analyze():

    with h5py.File(FILE_PATH, "r") as hdf:

        traces = hdf["traces"]
        peaks = []

        for start in range(0, len(traces), 256):
            batch = traces[start:start + 256]

            batch_peaks = np.max(
                np.abs(batch),
                axis=(1, 2),
            )

            peaks.extend(batch_peaks)

        peaks = np.asarray(peaks)

        print("\nAMPLITUDE DISTRIBUTION")
        print("Minimum:", np.min(peaks))
        print("Median:", np.median(peaks))
        print("90th percentile:", np.percentile(peaks, 90))
        print("95th percentile:", np.percentile(peaks, 95))
        print("99th percentile:", np.percentile(peaks, 99))
        print("99.9th percentile:", np.percentile(peaks, 99.9))
        print("Maximum:", np.max(peaks))

        extreme_indices = np.argsort(peaks)[-10:][::-1]

        print("\nTOP 10 EXTREME RECORDS")

        for index in extreme_indices:
            print(
                f"Record: {index} | "
                f"Peak amplitude: {peaks[index]:.2f}"
            )


if __name__ == "__main__":
    analyze()