import pandas as pd
import pytest

from seismopulse.data.metadata import (
    load_metadata,
    summarize_metadata,
)


@pytest.fixture
def sample_metadata():
    return pd.DataFrame({
        "trace_name": ["trace_001", "trace_002", "trace_003"],
        "trace_category": [
            "earthquake_local",
            "noise",
            "earthquake_local",
        ],
        "trace_p_arrival_sample": [120, None, 150],
        "trace_s_arrival_sample": [250, None, 300],
    })


def test_load_metadata_successfully(tmp_path, sample_metadata):
    csv_path = tmp_path / "metadata.csv"
    sample_metadata.to_csv(csv_path, index=False)

    result = load_metadata(csv_path)

    assert len(result) == 3
    assert "trace_name" in result.columns


def test_load_metadata_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_metadata(tmp_path / "missing.csv")


def test_load_metadata_rejects_empty_file(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("trace_name,trace_category\n")

    with pytest.raises(ValueError):
        load_metadata(csv_path)


def test_load_metadata_rejects_missing_columns(tmp_path):
    csv_path = tmp_path / "invalid.csv"

    pd.DataFrame({
        "trace_name": ["trace_001"],
    }).to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_metadata(csv_path)


def test_summarize_metadata_counts_categories(sample_metadata):
    summary = summarize_metadata(sample_metadata)

    assert summary["total_records"] == 3
    assert summary["category_counts"]["earthquake_local"] == 2
    assert summary["category_counts"]["noise"] == 1


def test_summarize_metadata_detects_duplicates():
    metadata = pd.DataFrame({
        "trace_name": ["trace_001", "trace_001"],
        "trace_category": ["noise", "noise"],
    })

    summary = summarize_metadata(metadata)

    assert summary["duplicate_trace_names"] == 1


def test_summarize_metadata_identifies_phase_columns(sample_metadata):
    summary = summarize_metadata(sample_metadata)

    assert "trace_p_arrival_sample" in summary["available_phase_columns"]
    assert "trace_s_arrival_sample" in summary["available_phase_columns"]


def test_summarize_metadata_rejects_invalid_input():
    with pytest.raises(TypeError):
        summarize_metadata("invalid")


def test_summarize_metadata_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        summarize_metadata(pd.DataFrame())