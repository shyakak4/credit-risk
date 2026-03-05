"""
src/features/build_features.py

Responsibility: Transform cleaned data into features ready for K-Means.
- Ordinal encoding for account levels (order matters)
- Binary encoding for sex
- One-hot encoding for purpose (no natural order)
- Ordinal encoding for housing
- RobustScaler for all numeric features (handles outliers)

MLOps principle: Feature engineering decisions come from EDA.
Every encoding choice is documented with its reason.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import RobustScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def encode_ordinal_saving_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Saving accounts has a natural order of financial strength.
    Ordinal encoding preserves this order for the clustering algorithm.

    none=0  →  little=1  →  moderate=2  →  quite rich=3  →  rich=4
    """
    order = {"none": 0, "little": 1, "moderate": 2, "quite rich": 3, "rich": 4}
    df["saving_accounts"] = df["saving_accounts"].map(order)
    logger.info("Encoded 'saving_accounts' as ordinal (0-4)")
    return df


def encode_ordinal_checking_account(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checking account also has a natural wealth order.

    none=0  →  little=1  →  moderate=2  →  rich=3
    """
    order = {"none": 0, "little": 1, "moderate": 2, "rich": 3}
    df["checking_account"] = df["checking_account"].map(order)
    logger.info("Encoded 'checking_account' as ordinal (0-3)")
    return df


def encode_ordinal_housing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Housing reflects financial stability.
    free (living with family/no cost) < rent < own

    free=0  →  rent=1  →  own=2
    """
    order = {"free": 0, "rent": 1, "own": 2}
    df["housing"] = df["housing"].map(order)
    logger.info("Encoded 'housing' as ordinal (0-2)")
    return df


def encode_binary_sex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary encoding. male=1, female=0.
    Simple and sufficient for two categories.
    """
    df["sex"] = (df["sex"] == "male").astype(int)
    logger.info("Encoded 'sex' as binary (male=1, female=0)")
    return df


def encode_onehot_purpose(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose has no natural order (car is not 'more' than education).
    One-hot encoding treats each category independently.

    We drop_first=False to keep all categories visible to the model.
    We prefix with 'purpose_' so column names stay clear.
    """
    purpose_dummies = pd.get_dummies(df["purpose"], prefix="purpose").astype(int)
    df = pd.concat([df.drop(columns=["purpose"]), purpose_dummies], axis=1)
    logger.info(f"One-hot encoded 'purpose' -> {list(purpose_dummies.columns)}")
    return df


def scale_features(df: pd.DataFrame, scaler_path: str) -> pd.DataFrame:
    """
    EDA showed significant outliers in credit_amount and duration.
    RobustScaler uses median and IQR instead of mean and std —
    making it resistant to outliers unlike StandardScaler.

    We save the fitted scaler so we can apply the SAME transformation
    to new data at prediction time. This is critical for correctness.
    """
    # All columns are now numeric at this point
    scaler = RobustScaler()
    scaled_array = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_array, columns=df.columns)

    # Save scaler for use in prediction pipeline
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"RobustScaler fitted and saved to: {scaler_path}")

    return df_scaled


def run_feature_engineering() -> pd.DataFrame:
    """
    Main feature engineering function. Called by DVC pipeline.
    Reads from data/processed/, writes to data/features/.
    """
    config = load_config()
    processed_path = config["data"]["processed_path"]
    features_path = config["data"]["features_path"]
    scaler_path = config["model"]["scaler_path"]

    logger.info("─── Starting feature engineering ───")

    df = pd.read_csv(processed_path)
    logger.info(f"Loaded cleaned data: {df.shape}")

    # Apply encodings in order
    df = encode_ordinal_saving_accounts(df)
    df = encode_ordinal_checking_account(df)
    df = encode_ordinal_housing(df)
    df = encode_binary_sex(df)
    df = encode_onehot_purpose(df)

    # Verify all columns are now numeric
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric columns found after encoding: {non_numeric}")

    logger.info(f"All features numeric. Shape before scaling: {df.shape}")

    # Scale
    df_scaled = scale_features(df, scaler_path)

    # Save
    Path(features_path).parent.mkdir(parents=True, exist_ok=True)
    df_scaled.to_csv(features_path, index=False)
    logger.info(f"Feature matrix saved to: {features_path}")
    logger.info(f"Final feature columns: {list(df_scaled.columns)}")
    logger.info("─── Feature engineering complete ───")

    return df_scaled


if __name__ == "__main__":
    run_feature_engineering()