"""Quality analysis of the STEAD earthquake subset."""

import h5py
import numpy as np

FILE_PATH = "data/raw/stead/test.hdf5"


def analyze_dataset():

    with h5py.File(FILE_PATH, "r") as hdf:

        traces = hdf["traces"]
        p_arrivals = hdf["p_arrival"][:]
        s_arrivals = hdf["s_arrival"][:]

        total = len(traces)
        invalid_signals = 0
        invalid_phases = 0
        invalid_order = 0

        peak_amplitudes = []
        standard_deviations = []

        for start in range(0, total, 256):

            batch = traces[start:start + 256]

            for waveform in batch:

                if not np.isfinite(waveform).all():
                    invalid_signals += 1
                    continue

                peak_amplitudes.append(
                    float(np.max(np.abs(waveform)))
                )

                standard_deviations.append(
                    float(np.std(waveform))
                )

        for p, s in zip(p_arrivals, s_arrivals):

            if not (0 <= p < 6000 and 0 <= s < 6000):
                invalid_phases += 1
            elif p >= s:
                invalid_order += 1

        print("\n========== STEAD QUALITY REPORT ==========")

        print("Total records:", total)
        print("Invalid waveforms:", invalid_signals)
        print("Invalid phase annotations:", invalid_phases)
        print("Incorrect P/S ordering:", invalid_order)

        print("\nAmplitude statistics:")
        print("Mean peak amplitude:", np.mean(peak_amplitudes))
        print("Maximum peak amplitude:", np.max(peak_amplitudes))
        print("Mean standard deviation:", np.mean(standard_deviations))

        print("\n========== ANALYSIS COMPLETE ==========")


if __name__ == "__main__":
    analyze_dataset()