"""
tests/test_features.py

Tests for src/features/build_features.py
Each encoding function is tested with known inputs and expected outputs.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    encode_binary_sex,
    encode_onehot_purpose,
    encode_ordinal_checking_account,
    encode_ordinal_housing,
    encode_ordinal_saving_accounts,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def cleaned_dataframe():
    """Cleaned DataFrame as it comes out of preprocessing."""
    return pd.DataFrame({
        "age": [25, 35, 45, 30],
        "sex": ["male", "female", "male", "female"],
        "job": [2, 1, 2, 3],
        "housing": ["own", "rent", "free", "own"],
        "saving_accounts": ["little", "none", "rich", "moderate"],
        "checking_account": ["none", "little", "rich", "moderate"],
        "credit_amount": [1000, 5000, 2500, 3000],
        "duration": [12, 24, 18, 36],
        "purpose": ["car", "education", "radio/TV", "business"],
    })


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_encode_ordinal_saving_accounts_correct_values(cleaned_dataframe):
    """Saving account levels should map to correct ordinal numbers."""
    result = encode_ordinal_saving_accounts(cleaned_dataframe.copy())
    # little=1, none=0, rich=4, moderate=2
    expected = [1, 0, 4, 2]
    assert list(result["saving_accounts"]) == expected


def test_encode_ordinal_saving_accounts_order_is_preserved(cleaned_dataframe):
    """Higher wealth level should always have higher numeric value."""
    result = encode_ordinal_saving_accounts(cleaned_dataframe.copy())
    none_val = result[cleaned_dataframe["saving_accounts"] == "none"]["saving_accounts"].iloc[0]
    little_val = result[cleaned_dataframe["saving_accounts"] == "little"]["saving_accounts"].iloc[0]
    rich_val = result[cleaned_dataframe["saving_accounts"] == "rich"]["saving_accounts"].iloc[0]
    assert none_val < little_val < rich_val


def test_encode_ordinal_checking_account_correct_values(cleaned_dataframe):
    """Checking account levels should map to correct ordinal numbers."""
    result = encode_ordinal_checking_account(cleaned_dataframe.copy())
    # none=0, little=1, rich=3, moderate=2
    expected = [0, 1, 3, 2]
    assert list(result["checking_account"]) == expected


def test_encode_ordinal_housing_correct_values(cleaned_dataframe):
    """Housing types should map to correct ordinal numbers."""
    result = encode_ordinal_housing(cleaned_dataframe.copy())
    # own=2, rent=1, free=0, own=2
    expected = [2, 1, 0, 2]
    assert list(result["housing"]) == expected


def test_encode_binary_sex_male_is_one(cleaned_dataframe):
    """Male should encode to 1."""
    result = encode_binary_sex(cleaned_dataframe.copy())
    male_rows = cleaned_dataframe["sex"] == "male"
    assert all(result.loc[male_rows, "sex"] == 1)


def test_encode_binary_sex_female_is_zero(cleaned_dataframe):
    """Female should encode to 0."""
    result = encode_binary_sex(cleaned_dataframe.copy())
    female_rows = cleaned_dataframe["sex"] == "female"
    assert all(result.loc[female_rows, "sex"] == 0)


def test_encode_binary_sex_only_zeros_and_ones(cleaned_dataframe):
    """After encoding, sex column should contain only 0 and 1."""
    result = encode_binary_sex(cleaned_dataframe.copy())
    assert set(result["sex"].unique()).issubset({0, 1})


def test_encode_onehot_purpose_removes_original_column(cleaned_dataframe):
    """Original purpose column should be gone after one-hot encoding."""
    result = encode_onehot_purpose(cleaned_dataframe.copy())
    assert "purpose" not in result.columns


def test_encode_onehot_purpose_creates_prefixed_columns(cleaned_dataframe):
    """One-hot columns should be prefixed with 'purpose_'."""
    result = encode_onehot_purpose(cleaned_dataframe.copy())
    purpose_cols = [c for c in result.columns if c.startswith("purpose_")]
    assert len(purpose_cols) > 0


def test_encode_onehot_purpose_values_are_integers(cleaned_dataframe):
    """One-hot encoded columns should be integers not booleans."""
    result = encode_onehot_purpose(cleaned_dataframe.copy())
    purpose_cols = [c for c in result.columns if c.startswith("purpose_")]
    for col in purpose_cols:
        assert result[col].dtype in [np.int32, np.int64, np.int8]


def test_encode_onehot_purpose_exactly_one_hot_per_row(cleaned_dataframe):
    """Each row should have exactly one purpose column set to 1."""
    result = encode_onehot_purpose(cleaned_dataframe.copy())
    purpose_cols = [c for c in result.columns if c.startswith("purpose_")]
    row_sums = result[purpose_cols].sum(axis=1)
    assert all(row_sums == 1)