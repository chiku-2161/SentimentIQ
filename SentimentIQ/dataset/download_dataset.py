"""
Dataset Preparation and Download Script for SentimentIQ
Downloads the genuine Twitter US Airline Sentiment Dataset (CrowdFlower / Kaggle),
cleans and standardizes the schema, and saves to dataset/reviews.csv.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import requests

# Adjust path to import config
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import RAW_DATA_URL, DATASET_PATH, DATASET_DIR


def download_and_prepare_dataset(output_path: Path = DATASET_PATH) -> pd.DataFrame:
    """
    Downloads the genuine Twitter Airline Sentiment dataset, validates and standardizes it.
    
    Args:
        output_path: Destination path for the prepared CSV file.
        
    Returns:
        Cleaned pandas DataFrame containing standard sentiment and review_text columns.
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"[Dataset Preparation] Fetching dataset from {RAW_DATA_URL}...")
    try:
        response = requests.get(RAW_DATA_URL, timeout=30)
        response.raise_for_status()
        raw_csv_path = DATASET_DIR / "raw_tweets.csv"
        with open(raw_csv_path, "wb") as f:
            f.write(response.content)
        print(f"[Dataset Preparation] Downloaded raw data ({len(response.content):,} bytes).")
        df_raw = pd.read_csv(raw_csv_path)
    except Exception as exc:
        print(f"[Warning] Direct download failed ({exc}). Checking if raw file or fallback exists...")
        raw_csv_path = DATASET_DIR / "raw_tweets.csv"
        if raw_csv_path.exists():
            df_raw = pd.read_csv(raw_csv_path)
        else:
            raise RuntimeError(
                f"Unable to download dataset and no local cache exists at {raw_csv_path}. Error: {exc}"
            )

    # Standardize column names
    print("[Dataset Preparation] Standardizing and cleaning dataset columns...")
    
    # Filter required columns
    required_cols = ["text", "airline_sentiment"]
    for col in required_cols:
        if col not in df_raw.columns:
            raise ValueError(f"Expected column '{col}' not found in raw dataset.")

    df_clean = df_raw.copy()
    
    # Rename columns to standard names
    df_clean = df_clean.rename(columns={
        "text": "review_text",
        "airline_sentiment": "sentiment"
    })
    
    # Optional metadata retention
    keep_columns = ["review_text", "sentiment"]
    if "airline" in df_clean.columns:
        keep_columns.append("airline")
    if "airline_sentiment_confidence" in df_clean.columns:
        df_clean["confidence"] = df_clean["airline_sentiment_confidence"]
        keep_columns.append("confidence")

    df_clean = df_clean[keep_columns]

    # Clean missing values and invalid rows
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=["review_text", "sentiment"])
    df_clean["review_text"] = df_clean["review_text"].astype(str).str.strip()
    df_clean["sentiment"] = df_clean["sentiment"].astype(str).str.strip().str.lower()
    
    # Retain only valid sentiment classes
    valid_classes = {"positive", "negative", "neutral"}
    df_clean = df_clean[df_clean["sentiment"].isin(valid_classes)]
    df_clean = df_clean[df_clean["review_text"].str.len() > 0]
    
    # Remove exact duplicate review texts
    df_clean = df_clean.drop_duplicates(subset=["review_text"])

    print(f"[Dataset Preparation] Processed rows: {initial_count} -> {len(df_clean)} valid entries.")
    print("\nClass Distribution:")
    print(df_clean["sentiment"].value_counts())
    print("\nClass Percentages:")
    print((df_clean["sentiment"].value_counts(normalize=True) * 100).round(2).astype(str) + "%")

    # Save standardized CSV
    df_clean.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n[Dataset Preparation] Successfully saved clean dataset to: {output_path}")

    # Clean temporary raw file if present
    if raw_csv_path.exists():
        try:
            os.remove(raw_csv_path)
        except OSError:
            pass

    return df_clean


if __name__ == "__main__":
    download_and_prepare_dataset()
