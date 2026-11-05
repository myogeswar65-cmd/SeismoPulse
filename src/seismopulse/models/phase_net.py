"""PhaseNet-inspired 1D U-Net for seismic phase picking."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two convolution layers with batch normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PhaseNet(nn.Module):
    """Encoder-decoder network for P/S seismic phase picking."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        base_channels: int = 32,
    ):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(
            base_channels * 4,
            base_channels * 8,
        )

        # Decoder
        self.up3 = nn.ConvTranspose1d(
            base_channels * 8,
            base_channels * 4,
            kernel_size=2,
            stride=2,
        )
        self.dec3 = ConvBlock(
            base_channels * 8,
            base_channels * 4,
        )

        self.up2 = nn.ConvTranspose1d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = ConvBlock(
            base_channels * 4,
            base_channels * 2,
        )

        self.up1 = nn.ConvTranspose1d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = ConvBlock(
            base_channels * 2,
            base_channels,
        )

        self.output = nn.Conv1d(
            base_channels,
            num_classes,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Per-sample class logits
        return self.output(d1)