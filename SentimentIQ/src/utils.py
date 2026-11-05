"""
SentimentIQ - Utility and Validation Module
Implements safe CSV parsing, file size validation, schema verification,
and error handling utilities for files and streams.
"""

import io
from pathlib import Path
from typing import Tuple, Optional, List, Union
import pandas as pd

from config.settings import MAX_FILE_SIZE_MB, SUPPORTED_FILE_TYPES


class FileValidationError(Exception):
    """Custom exception raised when an uploaded file fails validation."""
    pass


def validate_file_size(
    file_bytes: bytes,
    max_mb: int = MAX_FILE_SIZE_MB
) -> None:
    """
    Validates that the file payload does not exceed the allowed size limit.
    
    Args:
        file_bytes: Raw bytes of uploaded file.
        max_mb: Maximum allowed megabytes.
        
    Raises:
        FileValidationError: If file exceeds size threshold.
    """
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        raise FileValidationError(
            f"File size ({size_mb:.2f} MB) exceeds maximum permitted limit of {max_mb} MB."
        )


def load_and_validate_csv(
    file_source: Union[str, Path, io.BytesIO, bytes],
    required_columns: Optional[List[str]] = None,
    max_mb: int = MAX_FILE_SIZE_MB
) -> pd.DataFrame:
    """
    Safely loads and validates a CSV file from path, bytes, or file-like buffer.
    
    Args:
        file_source: File path, BytesIO stream, or raw bytes.
        required_columns: Optional list of column names that must exist in CSV.
        max_mb: Maximum size in MB.
        
    Returns:
        Validated pandas DataFrame.
        
    Raises:
        FileValidationError: If parsing fails, file is empty, or columns are missing.
    """
    # 1. Size check if bytes
    if isinstance(file_source, bytes):
        validate_file_size(file_source, max_mb)
        buffer = io.BytesIO(file_source)
    elif isinstance(file_source, io.BytesIO):
        file_bytes = file_source.getvalue()
        validate_file_size(file_bytes, max_mb)
        file_source.seek(0)
        buffer = file_source
    elif isinstance(file_source, (str, Path)):
        path = Path(file_source)
        if not path.exists():
            raise FileValidationError(f"File not found at: {path}")
        if path.suffix.lower() not in SUPPORTED_FILE_TYPES:
            raise FileValidationError(
                f"Unsupported file format '{path.suffix}'. Only .csv files are supported."
            )
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_mb:
            raise FileValidationError(
                f"File size ({size_mb:.2f} MB) exceeds maximum permitted limit of {max_mb} MB."
            )
        buffer = path
    else:
        raise FileValidationError(f"Invalid file source type: {type(file_source)}")

    # 2. Check for empty payload
    if isinstance(buffer, io.BytesIO) and len(buffer.getvalue().strip()) == 0:
        raise FileValidationError("Uploaded CSV file is completely empty.")
    if isinstance(buffer, Path) and buffer.stat().st_size == 0:
        raise FileValidationError("Uploaded CSV file is completely empty.")

    # 3. Multi-encoding robust CSV loading
    df = None
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    last_error = None
    
    for enc in encodings:
        try:
            if isinstance(buffer, io.BytesIO):
                buffer.seek(0)
            df = pd.read_csv(buffer, encoding=enc)
            break
        except pd.errors.EmptyDataError:
            raise FileValidationError("Uploaded CSV file is completely empty.")
        except Exception as err:
            last_error = err
            continue

    if df is None:
        raise FileValidationError(
            f"Failed to parse CSV file. Ensure valid CSV encoding. Details: {last_error}"
        )

    # 3. Content and row validation
    if df.empty:
        raise FileValidationError("Uploaded CSV file is completely empty.")

    if len(df.columns) == 0:
        raise FileValidationError("CSV contains no valid header columns.")

    # 4. Required columns check
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise FileValidationError(
                f"Missing required column(s): {', '.join(missing)}. "
                f"Available columns are: {', '.join(df.columns)}"
            )

    return df


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    """
    Intelligently suggests the most likely text column containing customer reviews.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Best candidate column name, or first object column, or None.
    """
    candidates = [
        "review_text", "text", "review", "reviews", "feedback", "comment",
        "comments", "tweet", "tweets", "message", "content", "body"
    ]
    lower_map = {col.lower(): col for col in df.columns}
    
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
            
    # Fallback to the object/string column with the highest mean character length
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(str_cols) > 0:
        lens = {col: df[col].astype(str).str.len().mean() for col in str_cols}
        return max(lens, key=lens.get)

    return None
