"""
SentimentIQ - Model Evaluation Module
Calculates performance metrics, confusion matrices, and classification reports.
Provides structured metric exports and visualization utilities.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from config.settings import SENTIMENT_CLASSES, METRICS_PATH


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = SENTIMENT_CLASSES
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics on held-out test predictions.
    
    Args:
        y_true: Ground truth target labels.
        y_pred: Model predicted labels.
        labels: List of class labels in deterministic order.
        
    Returns:
        Dictionary containing overall and per-class metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    
    # Macro metrics (unweighted average across classes, sensitive to minority classes)
    precision_macro = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    
    # Weighted metrics (proportional to class frequencies)
    precision_weighted = float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    recall_weighted = float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    
    # Full classification report dictionary
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )
    
    # Format per-class breakdown
    per_class = {}
    for cls in labels:
        if cls in report:
            per_class[cls] = {
                "precision": round(float(report[cls]["precision"]), 4),
                "recall": round(float(report[cls]["recall"]), 4),
                "f1_score": round(float(report[cls]["f1-score"]), 4),
                "support": int(report[cls]["support"]),
            }

    metrics = {
        "overall": {
            "accuracy": round(acc, 4),
            "precision_macro": round(precision_macro, 4),
            "recall_macro": round(recall_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "precision_weighted": round(precision_weighted, 4),
            "recall_weighted": round(recall_weighted, 4),
            "f1_weighted": round(f1_weighted, 4),
            "total_test_samples": int(len(y_true)),
        },
        "labels": labels,
        "confusion_matrix": cm,
        "per_class": per_class,
    }
    
    return metrics


def save_metrics(metrics: Dict[str, Any], filepath: Path = METRICS_PATH) -> None:
    """
    Saves evaluation metrics dictionary to a JSON file.
    
    Args:
        metrics: Dictionary of evaluated metrics.
        filepath: Destination file path.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[Evaluation] Metrics saved to {filepath}")


def load_metrics(filepath: Path = METRICS_PATH) -> Dict[str, Any]:
    """
    Loads saved evaluation metrics from JSON.
    
    Args:
        filepath: Path to the metrics JSON file.
        
    Returns:
        Dictionary of stored evaluation metrics.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Metrics file not found at {filepath}. Please train the model first by running train_model.py."
        )
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
