
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.model_service import seismopulse_service

FILE = ROOT / "experiments" / "validation_waveform.json"

with open(FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

waveform = data["waveform"] if isinstance(data, dict) and "waveform" in data else data

sample_rate = (
    data.get("sample_rate_hz", 100)
    if isinstance(data, dict)
    else 100
)

print("Loading model...")
seismopulse_service.load_model()

print("\nWaveform information:")
print("Shape:", np.array(waveform).shape)
print("Sample rate:", sample_rate)

print("\nRunning fresh model inference...")

result = seismopulse_service.predict(
    waveform=waveform,
    sample_rate_hz=sample_rate
)

print("\n========== P-WAVE ==========")
print(result["p_wave"])

print("\n========== S-WAVE ==========")
print(result["s_wave"])

print("\n========== STATUS ==========")
print(result["status"])
