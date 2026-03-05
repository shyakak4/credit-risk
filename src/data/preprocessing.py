"""
src/data/preprocessing.py

Responsibility: Clean raw data based on EDA findings.
- Fill missing values correctly
- Remove the unnamed index column
- Standardize column names
- Basic type corrections

MLOps principle: Every decision here is justified by EDA.
No guessing. No magic numbers without explanation.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def drop_unnamed_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    The raw CSV has an unnamed index column from the original source.
    We drop it — it carries no information.
    """
    unnamed_cols = [c for c in df.columns if "unnamed" in c.lower()]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        logger.info(f"Dropped unnamed index columns: {unnamed_cols}")
    return df


def fill_missing_account_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    EDA finding: 'Saving accounts' (18.3%) and 'Checking account' (39.4%)
    have missing values.

    These are NOT random missing — they mean the customer has NO account
    of that type. So we fill with 'none', not with mode or mean.
    Filling with mode would destroy the most important risk signal.
    """
    for col in ["Saving accounts", "Checking account"]:
        before = df[col].isnull().sum()
        df[col] = df[col].fillna("none")
        logger.info(f"'{col}': filled {before} missing values with 'none'")
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to snake_case for consistency across the pipeline.
    Python variables and model features should never have spaces.
    """
    rename_map = {
        "Age": "age",
        "Sex": "sex",
        "Job": "job",
        "Housing": "housing",
        "Saving accounts": "saving_accounts",
        "Checking account": "checking_account",
        "Credit amount": "credit_amount",
        "Duration": "duration",
        "Purpose": "purpose",
    }
    df = df.rename(columns=rename_map)
    logger.info(f"Columns renamed to snake_case: {list(df.columns)}")
    return df


def validate_cleaned_data(df: pd.DataFrame) -> bool:
    """
    After cleaning, verify no unexpected missing values remain
    and shapes are as expected.
    """
    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls > 0:
        raise ValueError(
            f"Preprocessing failed — {remaining_nulls} null values remain:\n"
            f"{df.isnull().sum()[df.isnull().sum() > 0]}"
        )

    expected_cols = [
        "age", "sex", "job", "housing", "saving_accounts",
        "checking_account", "credit_amount", "duration", "purpose"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns after cleaning: {missing}")

    logger.info("Cleaned data validation passed ✓")
    logger.info(f"  Final shape: {df.shape}")
    return True


def run_preprocessing() -> pd.DataFrame:
    """
    Main preprocessing function. Called by DVC pipeline.
    Reads from data/raw/, writes to data/processed/.
    """
    config = load_config()
    raw_path = config["data"]["raw_path"]
    processed_path = config["data"]["processed_path"]

    logger.info("─── Starting preprocessing ───")

    # Load raw
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded raw data: {df.shape}")

    # Apply cleaning steps in order
    df = drop_unnamed_index(df)
    df = fill_missing_account_columns(df)
    df = standardize_column_names(df)

    # Validate result
    validate_cleaned_data(df)

    # Save
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info(f"Cleaned data saved to: {processed_path}")
    logger.info("─── Preprocessing complete ───")

    return df


if __name__ == "__main__":
    run_preprocessing()