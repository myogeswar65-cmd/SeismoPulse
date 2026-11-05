
"""Validate real SeismoPulse records and P/S phase target coverage."""

import torch

from seismopulse.data.training_dataset import SeismicPhaseDataset


def main():
    dataset = SeismicPhaseDataset(
        earthquake_path="data/raw/stead/train.hdf5",
        noise_path="data/raw/stead/train_noise.hdf5",
        earthquake_indices=[0, 1, 2],
        noise_indices=[0, 1, 2],
    )

    print("\n========== SEISMOPULSE PHASE TARGET CHECK ==========")

    try:
        for index in range(len(dataset)):
            sample = dataset[index]

            waveform = sample["waveform"]
            targets = sample["targets"]
            source = sample["source"]

            noise_mass = targets[0].sum().item()
            p_mass = targets[1].sum().item()
            s_mass = targets[2].sum().item()

            p_peak = targets[1].argmax().item()
            s_peak = targets[2].argmax().item()

            p_peak_value = targets[1].max().item()
            s_peak_value = targets[2].max().item()

            print(f"\nRecord {index + 1}: {source}")
            print(f"Metadata: {sample['metadata']}")
            print(f"Waveform shape: {tuple(waveform.shape)}")
            print(
                f"Target mass - Noise: {noise_mass:.2f}, "
                f"P: {p_mass:.2f}, S: {s_mass:.2f}"
            )

            if source == "earthquake":
                print(f"P peak: sample {p_peak}, value {p_peak_value:.4f}")
                print(f"S peak: sample {s_peak}, value {s_peak_value:.4f}")

                if p_mass == 0 or s_mass == 0:
                    print("WARNING: Missing P or S target in this record.")
                else:
                    print("P/S targets: PRESENT")

            else:
                print("Noise-only record")

            assert waveform.shape == (3, 6000)
            assert targets.shape == (3, 6000)
            assert torch.isfinite(waveform).all()
            assert torch.isfinite(targets).all()

            assert torch.allclose(
                targets.sum(dim=0),
                torch.ones(6000),
                atol=1e-5,
            )

    finally:
        dataset.close()

    print("\n========== CHECK COMPLETED ==========")


if __name__ == "__main__":
    main()
