import h5py
import numpy as np

FILE_PATH = "data/raw/stead/test.hdf5"

with h5py.File(FILE_PATH, "r") as hdf:

    print("\n========== FIRST RECORD ==========")

    metadata = hdf["metadata"][0]

    if isinstance(metadata, bytes):
        metadata = metadata.decode("utf-8")

    print("\nMetadata:")
    print(repr(metadata))

    print("\nP arrival:", hdf["p_arrival"][0])
    print("S arrival:", hdf["s_arrival"][0])

    waveform = hdf["traces"][0]

    print("\nWaveform shape:", waveform.shape)
    print("Waveform dtype:", waveform.dtype)
    print("Minimum:", np.min(waveform))
    print("Maximum:", np.max(waveform))
    print("Mean:", np.mean(waveform))
    print("Standard deviation:", np.std(waveform))

    print("\n========== COMPLETE ==========")