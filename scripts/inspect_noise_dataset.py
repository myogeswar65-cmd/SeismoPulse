import h5py
import numpy as np

path = "data/raw/stead/train_noise.hdf5"

with h5py.File(path, "r") as hdf:
    print("Root keys:", list(hdf.keys()))

    for key in hdf.keys():
        item = hdf[key]
        print(f"{key}: shape={item.shape}, dtype={item.dtype}")

    traces = hdf["traces"]
    waveform = np.asarray(traces[0], dtype=np.float32)

    print("\nFirst noise waveform:")
    print("Shape:", waveform.shape)
    print("Min:", waveform.min())
    print("Max:", waveform.max())
    print("Mean:", waveform.mean())
    print("Std:", waveform.std())
    print("Finite:", np.isfinite(waveform).all())
