"""
tests/test_preprocessing.py

Tests for src/data/preprocessing.py
Each cleaning function is tested independently.
"""

import pandas as pd
import pytest

from src.data.preprocessing import (
    drop_unnamed_index,
    fill_missing_account_columns,
    standardize_column_names,
    validate_cleaned_data,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_dataframe():
    """Simulates what comes out of the raw CSV including unnamed index."""
    return pd.DataFrame({
        "Unnamed: 0": [0, 1, 2],
        "Age": [25, 35, 45],
        "Sex": ["male", "female", "male"],
        "Job": [2, 1, 2],
        "Housing": ["own", "rent", "free"],
        "Saving accounts": ["little", None, "moderate"],
        "Checking account": [None, "little", None],
        "Credit amount": [1000, 5000, 2500],
        "Duration": [12, 24, 18],
        "Purpose": ["car", "education", "radio/TV"],
    })


@pytest.fixture
def cleaned_dataframe():
    """Simulates fully cleaned data with snake_case columns."""
    return pd.DataFrame({
        "age": [25, 35, 45],
        "sex": ["male", "female", "male"],
        "job": [2, 1, 2],
        "housing": ["own", "rent", "free"],
        "saving_accounts": ["little", "none", "moderate"],
        "checking_account": ["none", "little", "none"],
        "credit_amount": [1000, 5000, 2500],
        "duration": [12, 24, 18],
        "purpose": ["car", "education", "radio/TV"],
    })


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_drop_unnamed_index_removes_column(raw_dataframe):
    """Unnamed index column should be removed."""
    result = drop_unnamed_index(raw_dataframe)
    assert "Unnamed: 0" not in result.columns


def test_drop_unnamed_index_keeps_other_columns(raw_dataframe):
    """All other columns should remain after dropping unnamed."""
    result = drop_unnamed_index(raw_dataframe)
    assert "Age" in result.columns
    assert "Credit amount" in result.columns


def test_drop_unnamed_index_does_nothing_if_no_unnamed(cleaned_dataframe):
    """If no unnamed column exists, DataFrame should be unchanged."""
    original_cols = list(cleaned_dataframe.columns)
    result = drop_unnamed_index(cleaned_dataframe)
    assert list(result.columns) == original_cols


def test_fill_missing_account_columns_fills_with_none(raw_dataframe):
    """NaN in Saving/Checking accounts should become the string 'none'."""
    result = fill_missing_account_columns(raw_dataframe)
    assert result["Saving accounts"].isnull().sum() == 0
    assert result["Checking account"].isnull().sum() == 0
    assert "none" in result["Saving accounts"].values
    assert "none" in result["Checking account"].values


def test_fill_missing_account_columns_preserves_existing_values(raw_dataframe):
    """Existing non-null values should not be changed."""
    result = fill_missing_account_columns(raw_dataframe)
    assert "little" in result["Saving accounts"].values
    assert "moderate" in result["Saving accounts"].values


def test_standardize_column_names_uses_snake_case(raw_dataframe):
    """All column names should be lowercase with underscores after renaming."""
    df = drop_unnamed_index(raw_dataframe)
    result = standardize_column_names(df)
    for col in result.columns:
        assert col == col.lower(), f"Column '{col}' is not lowercase"
        assert " " not in col, f"Column '{col}' contains a space"


def test_standardize_column_names_expected_outputs(raw_dataframe):
    """Spot check specific renames."""
    df = drop_unnamed_index(raw_dataframe)
    result = standardize_column_names(df)
    assert "credit_amount" in result.columns
    assert "saving_accounts" in result.columns
    assert "checking_account" in result.columns


def test_validate_cleaned_data_passes_on_clean_input(cleaned_dataframe):
    """Fully cleaned data should pass validation."""
    result = validate_cleaned_data(cleaned_dataframe)
    assert result is True


def test_validate_cleaned_data_fails_if_nulls_remain():
    """Cleaned data with remaining nulls should fail validation."""
    df_with_nulls = pd.DataFrame({
        "age": [25, None],
        "sex": ["male", "female"],
        "job": [2, 1],
        "housing": ["own", "rent"],
        "saving_accounts": ["little", "none"],
        "checking_account": ["none", "little"],
        "credit_amount": [1000, 5000],
        "duration": [12, 24],
        "purpose": ["car", "education"],
    })
    with pytest.raises(ValueError, match="null values remain"):
        validate_cleaned_data(df_with_nulls)