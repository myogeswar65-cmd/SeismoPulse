
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.seismopulse.models.phase_net import PhaseNet
from src.seismopulse.inference.phase_picker import (
    predict_phases,
    DEFAULT_P_THRESHOLD,
    DEFAULT_S_THRESHOLD,
)

HF_MODEL_REPO = os.getenv(
    "HF_MODEL_REPO",
    "MrYogi7/seismopulse-private-model"
)

HF_MODEL_FILENAME = os.getenv(
    "HF_MODEL_FILENAME",
    "phase_net_best.pt"
)


class SeismoPulseService:

    def __init__(self):
        self.model = None
        self.checkpoint_epoch = None

    def _resolve_checkpoint_path(self):
        # 1. Use an explicitly configured local model path.
        configured_path = os.getenv("SEISMOPULSE_MODEL_PATH")

        if configured_path:
            checkpoint_path = Path(configured_path)

            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Configured model checkpoint not found: {checkpoint_path}"
                )

            return checkpoint_path

        # 2. Download the private model from Hugging Face.
        hf_token = os.getenv("HF_TOKEN")

        if hf_token:
            print("Downloading private SeismoPulse model from Hugging Face...")

            checkpoint_path = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=HF_MODEL_FILENAME,
                token=hf_token
            )

            return Path(checkpoint_path)

        # 3. Fall back to the local checkpoint.
        local_checkpoint = (
            ROOT / "experiments" / "checkpoints" / "phase_net_best.pt"
        )

        if local_checkpoint.is_file():
            return local_checkpoint

        raise FileNotFoundError(
            "SeismoPulse model checkpoint is unavailable. "
            "Configure SEISMOPULSE_MODEL_PATH or HF_TOKEN."
        )

    def load_model(self):
        checkpoint_path = self._resolve_checkpoint_path()

        print(f"Loading SeismoPulse checkpoint: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
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
