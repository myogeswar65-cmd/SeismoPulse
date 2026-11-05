
import math

from pydantic import BaseModel, Field, validator


MAX_SAMPLES = 200_000
MIN_SAMPLES = 100


class PredictionRequest(BaseModel):
    waveform: list[list[float]] = Field(
        ...,
        description="Three-component seismic waveform: (samples, 3) or (3, samples)"
    )

    sample_rate_hz: float = Field(
        default=100.0,
        gt=0,
        le=1000
    )

    @validator("waveform")
    def validate_waveform(cls, waveform):
        if not waveform:
            raise ValueError("Waveform cannot be empty.")

        if len(waveform) < 3:
            raise ValueError("Waveform must contain three seismic components.")

        if not all(isinstance(row, list) for row in waveform):
            raise ValueError("Waveform must be a two-dimensional array.")

        row_lengths = [len(row) for row in waveform]

        if len(set(row_lengths)) != 1:
            raise ValueError("Waveform rows must have equal lengths.")

        # Accept (samples, 3) or (3, samples).
        if len(waveform[0]) == 3:
            sample_count = len(waveform)
        elif len(waveform) == 3:
            sample_count = len(waveform[0])
        else:
            raise ValueError(
                "Invalid waveform shape. Expected (samples, 3) or (3, samples)."
            )

        if sample_count < MIN_SAMPLES:
            raise ValueError(
                f"Waveform must contain at least {MIN_SAMPLES} samples."
            )

        if sample_count > MAX_SAMPLES:
            raise ValueError(
                f"Waveform exceeds the maximum of {MAX_SAMPLES} samples."
            )

        for row in waveform:
            for value in row:
                if not math.isfinite(value):
                    raise ValueError(
                        "Waveform contains an invalid value. "
                        "All values must be finite numbers."
                    )

        return waveform
