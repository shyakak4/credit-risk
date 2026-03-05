"""
src/data/ingestion.py

Responsibility: Load and validate raw data from data/raw/.
Since the CSV is already placed in data/raw/ manually (or via DVC pull),
this script only validates it arrived correctly — no copying needed.

In a production system this is where you would pull from a database,
API, or cloud storage. For now it reads and validates the local file.
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
    """Load project configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}")
    return config


def load_raw_data(raw_path: str) -> pd.DataFrame:
    """
    Load raw CSV from data/raw/.
    Raises a clear error if the file is missing.
    """
    path = Path(raw_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at: {raw_path}\n"
            "Make sure you placed the CSV in data/raw/ before running the pipeline."
        )

    df = pd.read_csv(path)
    logger.info(f"Raw data loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def validate_raw_data(df: pd.DataFrame, expected_columns: list) -> bool:
    """
    Sanity checks before data moves downstream.
    Catches problems early so preprocessing never runs on bad data.
    """
    # Check 1: not empty
    if df.empty:
        raise ValueError("Dataset is empty.")

    # Check 2: expected columns exist
    missing_cols = [c for c in expected_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # Check 3: not entirely duplicated
    duplicate_ratio = df.duplicated().sum() / len(df)
    if duplicate_ratio > 0.9:
        raise ValueError("Over 90% duplicate rows — data may be corrupted.")

    logger.info("Raw data validation passed")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Duplicates: {df.duplicated().sum()}")

    # Report missing values — expected in this dataset, not an error
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.info("  Missing values (expected):")
        for col, count in missing.items():
            logger.info(f"    {col}: {count} ({count/len(df)*100:.1f}%)")

    return True


def run_ingestion() -> pd.DataFrame:
    """
    Main ingestion function. Called by DVC pipeline.
    """
    config = load_config()
    raw_path = config["data"]["raw_path"]
    expected_cols = config["data"]["expected_columns"]

    logger.info("--- Starting data ingestion ---")

    df = load_raw_data(raw_path)
    validate_raw_data(df, expected_cols)

    logger.info("--- Ingestion complete ---")
    return df


if __name__ == "__main__":
    run_ingestion()