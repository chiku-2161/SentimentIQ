"""
SentimentIQ - Analytics and Intelligence Module
Extracts n-gram word frequencies, sentiment patterns, and dataset summary statistics.
Generates exportable analytical reports for customer feedback intelligence.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from config.settings import (
    CLASS_POSITIVE,
    CLASS_NEGATIVE,
    CLASS_NEUTRAL,
    SENTIMENT_CLASSES,
)
from src.preprocessing import clean_text_series

# Curated domain stopwords to remove common conversational fillers while preserving descriptive terms
ANALYTICS_STOPWORDS = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "their", "its", "our", "this", "that", "these",
    "those", "what", "which", "who", "whom", "flight", "plane", "airline",
    "get", "got", "just", "now", "will", "would", "could", "should", "can",
    "co", "http", "https", "rt", "via"
}


def compute_sentiment_distribution(
    df: pd.DataFrame,
    sentiment_column: str = "sentiment"
) -> Dict[str, Any]:
    """
    Computes absolute counts and relative percentages for sentiment classes.
    
    Args:
        df: Input DataFrame.
        sentiment_column: Name of the column with sentiment classes.
        
    Returns:
        Dictionary with count and percentage metrics.
    """
    if sentiment_column not in df.columns:
        raise ValueError(f"Column '{sentiment_column}' not found in DataFrame.")

    total = len(df)
    counts = df[sentiment_column].value_counts().to_dict()
    
    distribution = {}
    for cls in SENTIMENT_CLASSES:
        cnt = counts.get(cls, 0)
        pct = round((cnt / total * 100), 2) if total > 0 else 0.0
        distribution[cls] = {
            "count": int(cnt),
            "percentage": float(pct)
        }

    return {
        "total_reviews": int(total),
        "distribution": distribution
    }


def extract_top_ngrams(
    texts: pd.Series,
    ngram_range: Tuple[int, int] = (1, 2),
    top_n: int = 15,
    min_df: int = 2
) -> List[Dict[str, Any]]:
    """
    Extracts top frequent n-grams from a text series.
    
    NOTE: These terms represent purely descriptive co-occurrence patterns in feedback,
    and should NOT be construed as verified causal explanations.
    
    Args:
        texts: Series of clean review strings.
        ngram_range: Tuple (min_n, max_n).
        top_n: Number of top terms to return.
        min_df: Minimum document frequency threshold.
        
    Returns:
        List of dictionaries containing 'term' and 'frequency'.
    """
    clean_series = texts.dropna().astype(str).str.strip()
    valid_texts = clean_series[clean_series.str.len() > 0]
    
    if len(valid_texts) == 0:
        return []

    try:
        vec = CountVectorizer(
            ngram_range=ngram_range,
            stop_words=list(ANALYTICS_STOPWORDS),
            min_df=min_df,
            max_features=2000
        )
        dtm = vec.fit_transform(valid_texts)
        term_frequencies = np.asarray(dtm.sum(axis=0)).flatten()
        vocab = np.array(vec.get_feature_names_out())
        
        # Sort descending by frequency
        sorted_indices = np.argsort(term_frequencies)[::-1][:top_n]
        
        results = [
            {"term": str(vocab[i]), "frequency": int(term_frequencies[i])}
            for i in sorted_indices
            if term_frequencies[i] > 0
        ]
        return results
    except Exception:
        # Fallback if vocabulary is too sparse or empty
        return []


def generate_feedback_insights(
    df: pd.DataFrame,
    text_column: str = "review_text",
    sentiment_column: str = "sentiment",
    top_n: int = 15
) -> Dict[str, Any]:
    """
    Produces full comparative linguistic insights across sentiment classes.
    
    Args:
        df: Input DataFrame.
        text_column: Column name containing review text.
        sentiment_column: Column name containing sentiment labels.
        top_n: Top N n-grams to extract.
        
    Returns:
        Dictionary of analytics and top descriptive patterns.
    """
    dist = compute_sentiment_distribution(df, sentiment_column=sentiment_column)
    
    pos_mask = df[sentiment_column] == CLASS_POSITIVE
    neg_mask = df[sentiment_column] == CLASS_NEGATIVE
    neu_mask = df[sentiment_column] == CLASS_NEUTRAL
    
    # Text lengths
    df_temp = df.copy()
    df_temp["char_length"] = df_temp[text_column].astype(str).str.len()
    df_temp["word_count"] = df_temp[text_column].astype(str).str.split().str.len()

    # Clean text to remove handles, URLs, and noise for clean n-grams
    clean_series = clean_text_series(df[text_column])
    df_temp["cleaned_text"] = clean_series

    pos_ngrams = extract_top_ngrams(df_temp.loc[pos_mask, "cleaned_text"], top_n=top_n)
    neg_ngrams = extract_top_ngrams(df_temp.loc[neg_mask, "cleaned_text"], top_n=top_n)

    insights = {
        "overview": dist,
        "text_statistics": {
            "avg_char_length": round(float(df_temp["char_length"].mean()), 1),
            "avg_word_count": round(float(df_temp["word_count"].mean()), 1),
            "positive_avg_words": round(float(df_temp.loc[pos_mask, "word_count"].mean()), 1) if pos_mask.any() else 0.0,
            "negative_avg_words": round(float(df_temp.loc[neg_mask, "word_count"].mean()), 1) if neg_mask.any() else 0.0,
            "neutral_avg_words": round(float(df_temp.loc[neu_mask, "word_count"].mean()), 1) if neu_mask.any() else 0.0,
        },
        "descriptive_patterns": {
            "top_positive_terms": pos_ngrams,
            "top_negative_terms": neg_ngrams,
            "pattern_disclaimer": "Descriptive patterns represent observed n-gram frequencies in customer texts, not confirmed causal drivers."
        }
    }
    
    return insights
