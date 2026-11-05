"""Inspect earthquake training data and phase annotations."""

import h5py
import numpy as np

path = "data/raw/stead/train.hdf5"

with h5py.File(path, "r") as hdf:
    print("ROOT KEYS:", list(hdf.keys()))

    for key in hdf.keys():
        item = hdf[key]
        print(
            f"{key}: shape={item.shape}, dtype={item.dtype}"
        )

    required = {"metadata", "p_arrival", "s_arrival", "traces"}

    if not required.issubset(hdf.keys()):
        raise ValueError(
            f"Unexpected HDF5 structure. Missing: {required - set(hdf.keys())}"
        )

    p = np.asarray(hdf["p_arrival"][:])
    s = np.asarray(hdf["s_arrival"][:])

    samples = hdf["traces"].shape[1]

    print("\n--- DATASET SUMMARY ---")
    print("Total records:", len(p))
    print("Samples per trace:", samples)
    print("Components per trace:", hdf["traces"].shape[2])

    print("\n--- P ARRIVAL ---")
    print("Minimum:", p.min())
    print("Maximum:", p.max())
    print("Missing (-1):", np.sum(p == -1))
    print("Out of range:", np.sum((p < 0) | (p >= samples)))

    print("\n--- S ARRIVAL ---")
    print("Minimum:", s.min())
    print("Maximum:", s.max())
    print("Missing (-1):", np.sum(s == -1))
    print("Out of range:", np.sum((s < 0) | (s >= samples)))

    valid = (p >= 0) & (p < samples) & (s >= 0) & (s < samples)

    print("\n--- PHASE ORDER ---")
    print("Valid P/S pairs:", np.sum(valid))
    print("S before or equal to P:", np.sum(valid & (s <= p)))

    print("\n--- FIRST FIVE RECORDS ---")
    for i in range(min(5, len(p))):
        metadata = hdf["metadata"][i]
        if isinstance(metadata, bytes):
            metadata = metadata.decode("utf-8")

        print(
            f"{i}: trace={metadata}, P={p[i]}, S={s[i]}"
        )