"""
tests/test_ingestion.py

Tests for src/data/ingestion.py
Every function that can be tested independently gets its own test.
"""

import pandas as pd
import pytest

from src.data.ingestion import validate_raw_data


# ── Fixtures — reusable test data ─────────────────────────────────────────────

@pytest.fixture
def valid_dataframe():
    """A clean DataFrame that should pass all validation checks."""
    return pd.DataFrame({
        "Age": [25, 35, 45],
        "Sex": ["male", "female", "male"],
        "Job": [2, 1, 2],
        "Housing": ["own", "rent", "free"],
        "Saving accounts": ["little", None, "moderate"],
        "Checking account": [None, "little", "rich"],
        "Credit amount": [1000, 5000, 2500],
        "Duration": [12, 24, 18],
        "Purpose": ["car", "education", "radio/TV"],
    })


@pytest.fixture
def expected_columns():
    return [
        "Age", "Sex", "Job", "Housing",
        "Saving accounts", "Checking account",
        "Credit amount", "Duration", "Purpose"
    ]


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_validate_raw_data_passes_on_valid_input(valid_dataframe, expected_columns):
    """Valid data should pass without raising any exception."""
    result = validate_raw_data(valid_dataframe, expected_columns)
    assert result is True


def test_validate_raw_data_fails_on_empty_dataframe(expected_columns):
    """Empty DataFrame should raise ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="empty"):
        validate_raw_data(empty_df, expected_columns)


def test_validate_raw_data_fails_on_missing_columns(valid_dataframe):
    """If expected columns are missing, should raise ValueError."""
    columns_that_dont_exist = ["Age", "NonExistentColumn", "AnotherFakeColumn"]
    with pytest.raises(ValueError, match="Missing expected columns"):
        validate_raw_data(valid_dataframe, columns_that_dont_exist)


def test_validate_raw_data_fails_on_fully_duplicated_data(expected_columns):
    """Data that is >90% duplicates should raise ValueError."""
    single_row = {
        "Age": 25, "Sex": "male", "Job": 2, "Housing": "own",
        "Saving accounts": "little", "Checking account": "little",
        "Credit amount": 1000, "Duration": 12, "Purpose": "car"
    }
    # Create 100 identical rows — 100% duplicates
    duplicate_df = pd.DataFrame([single_row] * 100)
    with pytest.raises(ValueError, match="duplicate"):
        validate_raw_data(duplicate_df, expected_columns)


def test_validate_raw_data_allows_partial_missing_values(valid_dataframe, expected_columns):
    """
    Missing values in Saving/Checking accounts are expected and normal.
    Validation should still pass — missing values are NOT an error here.
    """
    result = validate_raw_data(valid_dataframe, expected_columns)
    assert result is True