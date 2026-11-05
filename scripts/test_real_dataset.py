from seismopulse.data.curated_stead import CuratedSTEADDataset

dataset = CuratedSTEADDataset("data/raw/stead/test.hdf5")

print("Total records:", len(dataset))

record = dataset[0]

print("Trace:", record["metadata"])
print("Waveform shape:", record["waveform"].shape)
print("Waveform dtype:", record["waveform"].dtype)
print("P arrival:", record["p_arrival"])
print("S arrival:", record["s_arrival"])