"""
SentimentIQ - Model Training Entrypoint
CLI script to trigger model training, feature extraction, evaluation, and artifact generation.

Usage:
    python train_model.py
    python train_model.py --dataset dataset/reviews.csv --test-size 0.20
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import DATASET_PATH, TEST_SIZE, RANDOM_STATE
from src.train import train_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the SentimentIQ TF-IDF + Logistic Regression Model."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to the training dataset CSV file (default: dataset/reviews.csv)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Proportion of the dataset to include in the test split (default: 0.20)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for reproducible stratified splitting (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        train_pipeline(
            dataset_path=args.dataset,
            test_size=args.test_size,
            random_state=args.random_state
        )
    except Exception as exc:
        print(f"\n[Error] Training failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
