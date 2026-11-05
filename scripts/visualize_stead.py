import h5py

from seismopulse.data.visualization import plot_waveform

FILE_PATH = "data/raw/stead/test.hdf5"

with h5py.File(FILE_PATH, "r") as hdf:
    waveform = hdf["traces"][0][:].T.copy()
    p_arrival = int(hdf["p_arrival"][0])
    s_arrival = int(hdf["s_arrival"][0])
    trace_name = hdf["metadata"][0]

plot_waveform(
    waveform=waveform,
    p_arrival=p_arrival,
    s_arrival=s_arrival,
    sampling_rate=100.0,
    title="STEAD Earthquake Waveform",
)