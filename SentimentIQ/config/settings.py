"""
SentimentIQ - Configuration and Settings
Centralizes application paths, model hyperparameters, visual theme colors, and validation constants.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# File Paths
RAW_DATA_URL = "https://raw.githubusercontent.com/ruchitgandhi/Twitter-Airline-Sentiment-Analysis/master/Tweets.csv"
DATASET_PATH = DATASET_DIR / "reviews.csv"
MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"

# Sentiment Classes and Encodings
CLASS_NEGATIVE = "negative"
CLASS_NEUTRAL = "neutral"
CLASS_POSITIVE = "positive"
SENTIMENT_CLASSES = [CLASS_NEGATIVE, CLASS_NEUTRAL, CLASS_POSITIVE]

# UI Color Palette
SENTIMENT_COLORS = {
    CLASS_POSITIVE: "#10B981",  # Emerald Green
    CLASS_NEUTRAL: "#6B7280",   # Slate Grey
    CLASS_NEGATIVE: "#EF4444",  # Crimson Red
}

# Training Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.20

TFIDF_CONFIG = {
    "ngram_range": (1, 2),
    "max_features": 10000,
    "sublinear_tf": True,
    "min_df": 2,
    "strip_accents": "unicode",
}

LOGISTIC_REGRESSION_CONFIG = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "solver": "lbfgs",
}

# Validation and Safety Settings
MAX_FILE_SIZE_MB = 10
SUPPORTED_FILE_TYPES = [".csv"]
MAX_PREVIEW_ROWS = 100
MAX_BATCH_ROWS = 50000
