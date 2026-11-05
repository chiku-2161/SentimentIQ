"""
Unit tests for text preprocessing and negation-preserving normalization.
"""

import pytest
import numpy as np
import pandas as pd

from src.preprocessing import clean_text, clean_text_series, expand_contractions


class TestTextPreprocessing:
    """Test suite covering text cleaning, edge cases, and normalization."""

    def test_lowercase_normalization(self):
        """Verify uppercase and mixed case text is normalized to lowercase."""
        text = "EXCELLENT SERVICE AND AWESOME STAFF!"
        expected = "excellent service and awesome staff!"
        assert clean_text(text) == expected

    def test_html_tag_removal(self):
        """Verify HTML tags and entities are stripped cleanly."""
        text = "<div><p>Great flight &amp; smooth landing.</p></div>"
        cleaned = clean_text(text)
        assert "<" not in cleaned and ">" not in cleaned
        assert "great flight" in cleaned and "smooth landing" in cleaned

    def test_url_removal(self):
        """Verify web URLs (http, https, www) are stripped."""
        text = "Check flight status at https://flightstatus.com/delay and www.airline.com for details"
        cleaned = clean_text(text)
        assert "https://" not in cleaned
        assert "www." not in cleaned
        assert "flightstatus" not in cleaned
        assert "airline.com" not in cleaned
        assert "check flight status at and for details" in cleaned

    def test_mention_removal(self):
        """Verify user handles (@mentions) are removed."""
        text = "@VirginAmerica @Delta customer support was very helpful."
        cleaned = clean_text(text)
        assert "@VirginAmerica" not in cleaned
        assert "@Delta" not in cleaned
        assert cleaned == "customer support was very helpful"

    def test_negation_preservation(self):
        """Verify critical negation words are retained and not stripped."""
        text1 = "This was not a good flight, never flying again."
        cleaned1 = clean_text(text1)
        assert "not" in cleaned1
        assert "never" in cleaned1

        text2 = "I can't recommend this airline, no customer care."
        cleaned2 = clean_text(text2)
        assert "not" in cleaned2 or "cannot" in cleaned2
        assert "no" in cleaned2

    def test_contraction_expansion(self):
        """Verify contractions are expanded into explicit negation forms."""
        expanded = expand_contractions("won't can't didn't isn't")
        assert "will not" in expanded
        assert "cannot" in expanded
        assert "did not" in expanded
        assert "is not" in expanded

    def test_whitespace_normalization(self):
        """Verify extra spaces, tabs, and newlines are collapsed."""
        text = "   Lots   of    \n\t  unnecessary    whitespace.   "
        assert clean_text(text) == "lots of unnecessary whitespace"

    def test_punctuation_normalization(self):
        """Verify repeated punctuation marks are collapsed to a single marker."""
        text = "Amazing experience!!!! Will travel again???"
        assert clean_text(text) == "amazing experience! will travel again?"

    def test_empty_and_whitespace_only(self):
        """Verify empty strings and whitespace return empty strings."""
        assert clean_text("") == ""
        assert clean_text("    ") == ""
        assert clean_text("\n\t") == ""

    def test_none_and_nan_inputs(self):
        """Verify None, np.nan, and float nulls do not crash and return empty strings."""
        assert clean_text(None) == ""
        assert clean_text(np.nan) == ""
        assert clean_text(float("nan")) == ""

    def test_non_string_inputs(self):
        """Verify numeric inputs are safely converted to string."""
        assert clean_text(12345) == "12345"

    def test_clean_text_series(self):
        """Verify vectorized Pandas series cleaning works across mixed types."""
        s = pd.Series(["<b>Great!</b>", None, "   ", "Not happy at all."])
        cleaned_series = clean_text_series(s)
        assert isinstance(cleaned_series, pd.Series)
        assert len(cleaned_series) == 4
        assert cleaned_series.iloc[0] == "great!"
        assert cleaned_series.iloc[1] == ""
        assert cleaned_series.iloc[2] == ""
        assert "not happy at all" in cleaned_series.iloc[3]
