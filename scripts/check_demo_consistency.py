
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
from pathlib import Path

from backend.app.services.model_service import seismopulse_service


ROOT = Path(__file__).resolve().parents[1]

DEMO_FILE = ROOT / "experiments" / "demo_validation_waveform.json"


def main():

    print("Loading validation demo...")

    with DEMO_FILE.open("r", encoding="utf-8") as f:
        demo = json.load(f)

    print("Loading current API model...")

    seismopulse_service.load_model()

    print("\nRunning fresh inference on EXACT demo waveform...")

    live = seismopulse_service.predict(
        waveform=demo["waveform"],
        sample_rate_hz=demo["sample_rate_hz"],
    )

    saved = demo["prediction"]

    print("\n========== SAVED DEMO PREDICTION ==========")

    print(
        "P confidence:",
        saved["p_wave"]["confidence"]
    )

    print(
        "S confidence:",
        saved["s_wave"]["confidence"]
    )

    print(
        "P arrival:",
        saved["p_wave"]["time_seconds"]
    )

    print(
        "S arrival:",
        saved["s_wave"]["time_seconds"]
    )

    print("\n========== CURRENT MODEL PREDICTION ==========")

    print(
        "P confidence:",
        live["p_wave"]["confidence"]
    )

    print(
        "S confidence:",
        live["s_wave"]["confidence"]
    )

    print(
        "P arrival:",
        live["p_wave"]["time_seconds"]
    )

    print(
        "S arrival:",
        live["s_wave"]["time_seconds"]
    )

    print("\n========== COMPARISON ==========")

    print(
        "P confidence difference:",
        abs(
            saved["p_wave"]["confidence"]
            - live["p_wave"]["confidence"]
        )
    )

    print(
        "S confidence difference:",
        abs(
            saved["s_wave"]["confidence"]
            - live["s_wave"]["confidence"]
        )
    )

    print(
        "P sample:",
        saved["p_wave"]["sample"],
        "vs",
        live["p_wave"]["sample"]
    )

    print(
        "S sample:",
        saved["s_wave"]["sample"],
        "vs",
        live["s_wave"]["sample"]
    )


if __name__ == "__main__":
    main()