"""
Unit tests for ClaimExtractor.
Tests sentence splitting and edge-case filtering behaviour.
"""
import pytest
from src.claim_extraction import ClaimExtractor


@pytest.fixture(scope="module")
def extractor():
    return ClaimExtractor()


def test_basic_extraction(extractor):
    text = "The company grew by 20%. The CEO is Jane Doe."
    claims = extractor.extract(text)
    assert len(claims) == 2
    assert claims[0] == "The company grew by 20%."
    assert claims[1] == "The CEO is Jane Doe."


def test_filters_short_fragments(extractor):
    """Strings of 5 chars or fewer should be dropped."""
    text = "Ok. The revenue declined by 5% last quarter."
    claims = extractor.extract(text)
    # "Ok." is only 3 chars — should be filtered out
    assert all(len(c) > 5 for c in claims)
    assert any("revenue" in c for c in claims)


def test_empty_string_returns_empty_list(extractor):
    assert extractor.extract("") == []


def test_single_sentence(extractor):
    text = "The Eiffel Tower is located in Paris."
    claims = extractor.extract(text)
    assert len(claims) == 1
    assert "Eiffel Tower" in claims[0]


def test_multisentence_paragraph(extractor):
    text = (
        "Acme Corp released its Q3 earnings report yesterday. "
        "The company revenue grew by 20% compared to last year. "
        "The current CEO, Jane Doe, stated that the growth was driven by the AI division. "
        "However, the hardware division saw a 5% decline in sales."
    )
    claims = extractor.extract(text)
    assert len(claims) == 4
