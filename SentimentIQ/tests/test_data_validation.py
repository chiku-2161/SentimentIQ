"""
Unit tests for CSV data validation, file size limits, and edge case detection.
"""

import io
import pytest
import pandas as pd

from src.utils import (
    load_and_validate_csv,
    validate_file_size,
    detect_text_column,
    FileValidationError,
)


class TestDataValidation:
    """Test suite covering file validation, schema enforcement, and security limits."""

    def test_file_size_within_limit(self):
        """Verify normal file size passes validation."""
        small_payload = b"col1,col2\nval1,val2\n"
        # Should not raise exception
        validate_file_size(small_payload, max_mb=10)

    def test_file_size_exceeded(self):
        """Verify oversized payloads raise FileValidationError."""
        # 1.5 MB payload against a 1 MB limit
        large_payload = b"x" * (1024 * 1024 + 500000)
        with pytest.raises(FileValidationError) as exc_info:
            validate_file_size(large_payload, max_mb=1)
        assert "exceeds maximum permitted limit" in str(exc_info.value)

    def test_valid_csv_stream_loading(self):
        """Verify valid CSV BytesIO stream is parsed correctly."""
        csv_bytes = b"review_id,feedback\n101,Great journey!\n102,Late arrival.\n"
        df = load_and_validate_csv(io.BytesIO(csv_bytes))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["review_id", "feedback"]

    def test_empty_csv_file(self):
        """Verify completely empty CSV files raise FileValidationError."""
        with pytest.raises(FileValidationError) as exc_info:
            load_and_validate_csv(io.BytesIO(b""))
        assert "empty" in str(exc_info.value).lower()

    def test_missing_required_columns(self):
        """Verify missing required column enforcement."""
        csv_bytes = b"user_id,rating\n1,5\n"
        with pytest.raises(FileValidationError) as exc_info:
            load_and_validate_csv(
                io.BytesIO(csv_bytes),
                required_columns=["review_text", "sentiment"]
            )
        assert "Missing required column(s)" in str(exc_info.value)

    def test_detect_text_column(self):
        """Verify automatic detection of review text column."""
        # Priority column naming
        df1 = pd.DataFrame({"id": [1], "review_text": ["Sample review"], "date": ["2025-01-01"]})
        assert detect_text_column(df1) == "review_text"

        df2 = pd.DataFrame({"id": [1], "feedback": ["Sample feedback"]})
        assert detect_text_column(df2) == "feedback"

        # Heuristic length fallback
        df3 = pd.DataFrame({"code": ["US"], "description": ["A very long detailed customer review text here"]})
        assert detect_text_column(df3) == "description"
