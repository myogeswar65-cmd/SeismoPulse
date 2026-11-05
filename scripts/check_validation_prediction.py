
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.seismopulse.data.build_loaders import build_dataloaders
from src.seismopulse.models.phase_net import PhaseNet
from src.seismopulse.inference.phase_picker import predict_phases

CHECKPOINT = ROOT / "experiments/checkpoints/phase_net_best.pt"

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

model = PhaseNet()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

_, loader = build_dataloaders(batch_size=4, num_workers=0)

for batch in loader:
    for i, source in enumerate(batch["source"]):
        if source != "earthquake":
            continue

        waveform = batch["waveform"][i]
        targets = batch["targets"][i]

        p_region = torch.where(targets[1] >= 0.5)[0]
        s_region = torch.where(targets[2] >= 0.5)[0]

        if len(p_region) == 0 or len(s_region) == 0:
            continue

        actual_p = int(p_region[len(p_region)//2])
        actual_s = int(s_region[len(s_region)//2])

        result = predict_phases(
            waveform=waveform.numpy(),
            model=model,
            sample_rate_hz=100
        )

        print("\nVALIDATION RECORD")
        print("Actual P:", actual_p, "samples")
        print("Actual S:", actual_s, "samples")
        print("Actual P time:", actual_p / 100, "seconds")
        print("Actual S time:", actual_s / 100, "seconds")
        print("\nModel prediction:")
        print(result)

        raise SystemExit(0)

raise RuntimeError("No earthquake validation record found.")
