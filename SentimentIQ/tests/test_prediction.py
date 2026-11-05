"""
Unit and integration tests for model loading, prediction routines, and output schemas.
"""

import pytest
import numpy as np
import pandas as pd

from config.settings import SENTIMENT_CLASSES, CLASS_POSITIVE, CLASS_NEGATIVE, CLASS_NEUTRAL
from src.predict import SentimentPredictor, get_predictor


@pytest.fixture(scope="module")
def predictor():
    """Loads predictor singleton once for module tests."""
    return get_predictor()


class TestSentimentPrediction:
    """Test suite for SentimentPredictor inference engine."""

    def test_model_loading(self, predictor):
        """Verify model and vectorizer load without error and have expected attributes."""
        assert predictor.model is not None
        assert predictor.vectorizer is not None
        assert set(predictor.classes_) == set(SENTIMENT_CLASSES)

    def test_single_prediction_output_structure(self, predictor):
        """Verify output dictionary contains all required fields and valid types."""
        result = predictor.predict_single("The flight attendants were remarkably friendly.")
        assert isinstance(result, dict)
        assert "sentiment" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "cleaned_text" in result
        assert "is_empty" in result
        assert result["sentiment"] in SENTIMENT_CLASSES
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["is_empty"] is False

    def test_probability_distribution_sum(self, predictor):
        """Verify predicted class probabilities sum to approximately 1.0."""
        result = predictor.predict_single("Average flight, nothing special happened.")
        probs = result["probabilities"]
        assert len(probs) == len(SENTIMENT_CLASSES)
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)

    def test_empty_input_handling(self, predictor):
        """Verify empty and whitespace inputs return graceful neutral fallback."""
        result = predictor.predict_single("   ")
        assert result["is_empty"] is True
        assert result["sentiment"] == CLASS_NEUTRAL
        assert result["confidence"] == pytest.approx(1.0 / len(SENTIMENT_CLASSES), abs=1e-3)

    def test_none_input_handling(self, predictor):
        """Verify None input does not throw an exception."""
        result = predictor.predict_single(None)
        assert result["is_empty"] is True
        assert result["sentiment"] == CLASS_NEUTRAL

    def test_positive_sentiment_inference(self, predictor):
        """Verify clear positive feedback is classified as positive."""
        text = "Absolutely wonderful experience! Best crew, clean cabin, and on time arrival."
        result = predictor.predict_single(text)
        assert result["sentiment"] == CLASS_POSITIVE
        assert result["confidence"] > 0.5

    def test_negative_sentiment_inference(self, predictor):
        """Verify clear negative feedback is classified as negative."""
        text = "Worst airline ever. Flight cancelled, lost my luggage, and terrible customer service."
        result = predictor.predict_single(text)
        assert result["sentiment"] == CLASS_NEGATIVE
        assert result["confidence"] > 0.5

    def test_batch_prediction_shape_and_types(self, predictor):
        """Verify batch prediction returns DataFrame with correct dimensions and columns."""
        texts = [
            "Loved the flight!",
            "Terrible delay and lost luggage.",
            "Gate 12 at Chicago O'Hare.",
            None,
            ""
        ]
        df_res = predictor.predict_batch(texts, include_probabilities=True)
        assert isinstance(df_res, pd.DataFrame)
        assert len(df_res) == len(texts)
        assert "predicted_sentiment" in df_res.columns
        assert "confidence" in df_res.columns
        assert "prob_positive" in df_res.columns
        assert "prob_neutral" in df_res.columns
        assert "prob_negative" in df_res.columns

        # Verify sentiments match allowed classes
        assert set(df_res["predicted_sentiment"].unique()).issubset(set(SENTIMENT_CLASSES))
