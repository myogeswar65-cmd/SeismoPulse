
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.seismopulse.models.phase_net import PhaseNet
from src.seismopulse.inference.phase_picker import (
    predict_phases,
    DEFAULT_P_THRESHOLD,
    DEFAULT_S_THRESHOLD,
)

# Use a configurable checkpoint path.
# If no environment variable is provided, use the local project checkpoint.
CHECKPOINT = Path(
    os.getenv(
        "SEISMOPULSE_MODEL_PATH",
        str(ROOT / "experiments" / "checkpoints" / "phase_net_best.pt")
    )
)


class SeismoPulseService:

    def __init__(self):
        self.model = None
        self.checkpoint_epoch = None

    def load_model(self):
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(
                "SeismoPulse model checkpoint is unavailable. "
                "Check the configured model path."
            )

        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=False
        )

        self.model = PhaseNet()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.checkpoint_epoch = checkpoint.get("epoch")

        torch.set_num_threads(min(4, torch.get_num_threads()))

        print(
            f"SeismoPulse model loaded successfully. "
            f"Checkpoint epoch: {self.checkpoint_epoch}"
        )

    def predict(self, waveform, sample_rate_hz=100):
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        return predict_phases(
            waveform=waveform,
            model=self.model,
            sample_rate_hz=sample_rate_hz,
            p_threshold=DEFAULT_P_THRESHOLD,
            s_threshold=DEFAULT_S_THRESHOLD
        )

    def model_info(self):
        return {
            "model": "SeismoPulse PhaseNet-inspired",
            "loaded": self.model is not None,
            "checkpoint_epoch": self.checkpoint_epoch,
            "device": "cpu",
            "sample_rate_hz": 100,
            "p_threshold": DEFAULT_P_THRESHOLD,
            "s_threshold": DEFAULT_S_THRESHOLD
        }


seismopulse_service = SeismoPulseService()
