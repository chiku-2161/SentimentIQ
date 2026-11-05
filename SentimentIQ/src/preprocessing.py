"""
SentimentIQ - Text Preprocessing Module
Implements reusable, production-ready text cleaning and normalization routines.
Preserves negation words and sentiment indicators while eliminating noise (HTML, URLs, mentions).
"""

import re
import html
from typing import Any, List, Union
import pandas as pd


# Contraction mapping to expand common negations and conversational shortcuts
CONTRACTIONS = {
    "won't": "will not",
    "can't": "cannot",
    "cannot": "cannot",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
    "dont": "do not",
    "didnt": "did not",
    "isnt": "is not",
    "arent": "are not",
    "wasnt": "was not",
    "werent": "were not",
    "wont": "will not",
    "cant": "cannot",
}

# Regex pre-compilation for maximum efficiency
RE_HTML = re.compile(r"<[^>]+?>")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MENTION = re.compile(r"@\w+")
RE_SPECIAL_CHARS = re.compile(r"[^a-zA-Z0-9\s!\?']")
RE_MULTI_WHITESPACE = re.compile(r"\s+")
RE_REPEATED_PUNCT = re.compile(r"([!?]){2,}")


def expand_contractions(text: str) -> str:
    """
    Expands conversational contractions to preserve semantic sentiment and negation.
    
    Args:
        text: Input string.
        
    Returns:
        String with expanded contractions.
    """
    words = text.split()
    expanded_words = []
    for word in words:
        lower_word = word.lower()
        if lower_word in CONTRACTIONS:
            expanded_words.append(CONTRACTIONS[lower_word])
        else:
            # Check suffix contraction like didn't or wasn't
            replaced = False
            for c_key, c_val in CONTRACTIONS.items():
                if lower_word.endswith(c_key) and len(lower_word) > len(c_key):
                    base = lower_word[:-len(c_key)]
                    expanded_words.append(f"{base} {c_val.strip()}")
                    replaced = True
                    break
            if not replaced:
                expanded_words.append(word)
    return " ".join(expanded_words)


def clean_text(text: Any) -> str:
    """
    Cleans and standardizes raw review text for sentiment modeling.
    
    Steps:
    1. Null / Non-string validation.
    2. Unescape HTML entities (e.g., &amp; -> &).
    3. Remove HTML tags.
    4. Remove web URLs.
    5. Remove user handles/mentions (@username) to avoid airline bias.
    6. Expand negations and contractions (e.g., aren't -> are not).
    7. Normalize punctuation while keeping informative markers (! and ?).
    8. Lowercase normalization.
    9. Collapse consecutive whitespace.
    
    Args:
        text: Raw text input (string, None, or float).
        
    Returns:
        Cleaned, normalized string.
    """
    if text is None or pd.isna(text):
        return ""
    
    if not isinstance(text, str):
        text = str(text)
        
    text = text.strip()
    if not text:
        return ""
    
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Strip HTML tags
    text = RE_HTML.sub(" ", text)
    
    # Remove web URLs
    text = RE_URL.sub(" ", text)
    
    # Remove @mentions (e.g., @VirginAmerica, @SouthwestAir)
    text = RE_MENTION.sub(" ", text)
    
    # Expand contractions and negations
    text = expand_contractions(text)
    
    # Remove noise characters while preserving alphanumeric, spaces, and expressive marks
    text = RE_SPECIAL_CHARS.sub(" ", text)
    
    # Normalize multiple punctuation (e.g., "!!!" -> "!")
    text = RE_REPEATED_PUNCT.sub(r"\1", text)
    
    # Lowercase normalization
    text = text.lower()
    
    # Collapse redundant whitespace
    text = RE_MULTI_WHITESPACE.sub(" ", text).strip()
    
    return text


def clean_text_series(series: pd.Series) -> pd.Series:
    """
    Vectorized cleaning for a pandas Series of text reviews.
    
    Args:
        series: Pandas Series containing text entries.
        
    Returns:
        Pandas Series containing cleaned text strings.
    """
    return series.fillna("").astype(str).map(clean_text)
