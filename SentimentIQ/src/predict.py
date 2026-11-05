"""
SentimentIQ - Prediction Engine
Provides robust single-text and batch inference routines with class probability scores,
resilient edge-case handling, and thread-safe artifact loading.
"""

from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import numpy as np
import pandas as pd
import joblib

from config.settings import (
    MODEL_PATH,
    VECTORIZER_PATH,
    CLASS_NEUTRAL,
    SENTIMENT_CLASSES,
)
from src.preprocessing import clean_text, clean_text_series


class SentimentPredictor:
    """
    Inference engine for SentimentIQ models.
    Loads and caches model and vectorizer weights to enable low-latency predictions.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        vectorizer_path: Path = VECTORIZER_PATH
    ):
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.model = None
        self.vectorizer = None
        self.classes_: List[str] = SENTIMENT_CLASSES
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads serialized model and vectorizer from disk safely."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Train the model using train_model.py first."
            )
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(
                f"Vectorizer file not found at {self.vectorizer_path}. Train the model using train_model.py first."
            )
            
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        
        # Ensure class labels align with model.classes_
        if hasattr(self.model, "classes_"):
            self.classes_ = list(self.model.classes_)

    def predict_single(self, text: Any) -> Dict[str, Any]:
        """
        Generates sentiment prediction and probability distribution for a single text input.
        
        Args:
            text: Raw input text.
            
        Returns:
            Dictionary with:
                - 'sentiment': 'positive', 'negative', or 'neutral'
                - 'confidence': float between 0.0 and 1.0
                - 'probabilities': Dict[str, float] for each class
                - 'cleaned_text': text after normalization
                - 'is_empty': bool indicating if input was empty/uninformative
        """
        cleaned = clean_text(text)
        
        # Edge case: Empty, whitespace, or eliminated string
        if not cleaned:
            # Default neutral with even probability spread
            even_prob = round(1.0 / len(self.classes_), 4)
            return {
                "sentiment": CLASS_NEUTRAL,
                "confidence": even_prob,
                "probabilities": {cls: even_prob for cls in self.classes_},
                "cleaned_text": "",
                "is_empty": True,
            }

        # Vectorize input
        X_vec = self.vectorizer.transform([cleaned])
        
        # Predict class probabilities
        probs = self.model.predict_proba(X_vec)[0]
        prob_dict = {
            cls: round(float(prob), 4)
            for cls, prob in zip(self.classes_, probs)
        }
        
        # Determine highest probability class
        pred_idx = int(np.argmax(probs))
        predicted_class = self.classes_[pred_idx]
        confidence = round(float(probs[pred_idx]), 4)
        
        return {
            "sentiment": predicted_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "cleaned_text": cleaned,
            "is_empty": False,
        }

    def predict_batch(
        self,
        texts: Union[List[str], pd.Series],
        include_probabilities: bool = True
    ) -> pd.DataFrame:
        """
        Processes a collection of text inputs efficiently via matrix vectorization.
        
        Args:
            texts: List or Series of raw review texts.
            include_probabilities: If True, adds individual class probability columns.
            
        Returns:
            DataFrame containing original texts, cleaned texts, predicted sentiment,
            confidence scores, and optional class probabilities.
        """
        if isinstance(texts, list):
            series = pd.Series(texts)
        else:
            series = texts.copy()
            
        cleaned_series = clean_text_series(series)
        
        # Identify non-empty rows for batch vectorization
        non_empty_mask = cleaned_series.str.strip().str.len() > 0
        
        # Initialize result containers
        sentiments = [CLASS_NEUTRAL] * len(series)
        confidences = [round(1.0 / len(self.classes_), 4)] * len(series)
        prob_matrix = np.full((len(series), len(self.classes_)), 1.0 / len(self.classes_))
        
        if non_empty_mask.any():
            valid_cleaned = cleaned_series[non_empty_mask].tolist()
            X_vec = self.vectorizer.transform(valid_cleaned)
            valid_probs = self.model.predict_proba(X_vec)
            valid_preds = self.model.predict(X_vec)
            valid_confs = np.max(valid_probs, axis=1)
            
            # Map back to full array
            valid_indices = np.where(non_empty_mask)[0]
            for i, idx in enumerate(valid_indices):
                sentiments[idx] = str(valid_preds[i])
                confidences[idx] = round(float(valid_confs[i]), 4)
                prob_matrix[idx] = valid_probs[i]
                
        results_df = pd.DataFrame({
            "original_text": series.values,
            "cleaned_text": cleaned_series.values,
            "predicted_sentiment": sentiments,
            "confidence": confidences,
        })
        
        if include_probabilities:
            for i, cls in enumerate(self.classes_):
                results_df[f"prob_{cls}"] = np.round(prob_matrix[:, i], 4)
                
        return results_df


# Module-level singleton helper
_PREDICTOR_INSTANCE: Optional[SentimentPredictor] = None


def get_predictor() -> SentimentPredictor:
    """Returns a singleton instance of SentimentPredictor to prevent redundant reloading."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = SentimentPredictor()
    return _PREDICTOR_INSTANCE
