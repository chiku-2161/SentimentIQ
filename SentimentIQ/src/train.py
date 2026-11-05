"""
SentimentIQ - Model Training Pipeline
Performs stratified splitting, TF-IDF vectorization, class-balanced Logistic Regression training,
leakage prevention, evaluation against held-out test data, and artifact persistence.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config.settings import (
    DATASET_PATH,
    MODEL_PATH,
    VECTORIZER_PATH,
    METRICS_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    TFIDF_CONFIG,
    LOGISTIC_REGRESSION_CONFIG,
    SENTIMENT_CLASSES,
)
from src.preprocessing import clean_text_series
from src.evaluation import compute_metrics, save_metrics


def train_pipeline(
    dataset_path: Path = DATASET_PATH,
    model_output_path: Path = MODEL_PATH,
    vectorizer_output_path: Path = VECTORIZER_PATH,
    metrics_output_path: Path = METRICS_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[LogisticRegression, TfidfVectorizer, Dict[str, Any]]:
    """
    Executes the end-to-end sentiment analysis training and evaluation pipeline.
    
    Args:
        dataset_path: Path to clean dataset CSV.
        model_output_path: Destination path for trained model artifact.
        vectorizer_output_path: Destination path for fitted TF-IDF artifact.
        metrics_output_path: Destination path for evaluation JSON metrics.
        test_size: Ratio for test split.
        random_state: Deterministic random seed.
        
    Returns:
        Tuple of (trained model, fitted vectorizer, evaluation metrics dictionary).
    """
    print("=" * 65)
    print("        SENTIMENTIQ - MACHINE LEARNING TRAINING PIPELINE        ")
    print("=" * 65)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run 'python dataset/download_dataset.py' first."
        )

    # 1. Load Data
    print(f"[1/6] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"      Total records loaded: {len(df):,}")

    if "review_text" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("Dataset must contain 'review_text' and 'sentiment' columns.")

    # 2. Text Preprocessing
    print("[2/6] Normalizing and preprocessing review text...")
    df["cleaned_text"] = clean_text_series(df["review_text"])
    
    # Filter out entries where text is empty after cleaning
    valid_mask = (df["cleaned_text"].str.strip().str.len() > 0) & (df["sentiment"].isin(SENTIMENT_CLASSES))
    df_clean = df[valid_mask].copy()
    print(f"      Valid entries after cleaning: {len(df_clean):,}")

    X = df_clean["cleaned_text"].values
    y = df_clean["sentiment"].values

    # 3. Stratified Train-Test Split (Prevent Data Leakage)
    print(f"[3/6] Performing Stratified Split ({int((1-test_size)*100)}% Train / {int(test_size*100)}% Test)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"      Training samples: {len(X_train_raw):,}")
    print(f"      Test samples:     {len(X_test_raw):,}")

    # 4. Feature Extraction: Fit TF-IDF ONLY on Training Data
    print("[4/6] Fitting TF-IDF Vectorizer (ngram range: 1-2, sublinear TF)...")
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    print(f"      Vocabulary size:  {len(vectorizer.vocabulary_):,} features")

    # 5. Model Training: Class-balanced Logistic Regression
    print("[5/6] Training Logistic Regression classifier with class balancing...")
    model = LogisticRegression(**LOGISTIC_REGRESSION_CONFIG)
    model.fit(X_train_vec, y_train)

    # 6. Evaluation on Held-Out Test Set
    print("[6/6] Evaluating on held-out test set...")
    y_pred = model.predict(X_test_vec)
    metrics = compute_metrics(y_true=y_test, y_pred=y_pred, labels=SENTIMENT_CLASSES)

    # Persist Artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)
    joblib.dump(vectorizer, vectorizer_output_path)
    save_metrics(metrics, metrics_output_path)

    # Print Summary Report
    ov = metrics["overall"]
    print("\n" + "-" * 65)
    print("                     EVALUATION RESULTS                         ")
    print("-" * 65)
    print(f"  Accuracy:           {ov['accuracy'] * 100:.2f}%")
    print(f"  Precision (Macro):  {ov['precision_macro'] * 100:.2f}%")
    print(f"  Recall (Macro):     {ov['recall_macro'] * 100:.2f}%")
    print(f"  F1-Score (Macro):   {ov['f1_macro'] * 100:.2f}%")
    print(f"  F1-Score (Weighted):{ov['f1_weighted'] * 100:.2f}%")
    print("-" * 65)
    print("  Per-Class Performance:")
    for cls, sc in metrics["per_class"].items():
        print(f"    - {cls.upper():<9}: Precision={sc['precision']:.3f} | Recall={sc['recall']:.3f} | F1={sc['f1_score']:.3f} (N={sc['support']})")
    print("-" * 65)
    print(f"[Done] Artifacts saved:")
    print(f"  - Model:      {model_output_path}")
    print(f"  - Vectorizer: {vectorizer_output_path}")
    print(f"  - Metrics:    {metrics_output_path}")
    print("=" * 65)

    return model, vectorizer, metrics


if __name__ == "__main__":
    train_pipeline()
