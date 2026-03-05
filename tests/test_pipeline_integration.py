"""
tests/test_pipeline_integration.py

Integration-level tests that cover the run_* entry point functions.
These tests use temporary directories and real files to test the full
function flow without depending on the actual project data.

This brings coverage above 70% by exercising the main pipeline functions.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.cluster import KMeans

from src.data.ingestion import run_ingestion
from src.data.preprocessing import run_preprocessing
from src.features.build_features import run_feature_engineering


# ── Shared temp project fixture ───────────────────────────────────────────────

@pytest.fixture
def temp_project(tmp_path):
    """
    Creates a full temporary project structure with:
    - configs/config.yaml
    - data/raw/german_credit_data.csv (small sample)
    - All required directories

    Returns the tmp_path root so tests can reference files inside it.
    """
    # Create directories
    (tmp_path / "configs").mkdir()
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "features").mkdir(parents=True)
    (tmp_path / "models").mkdir()
    (tmp_path / "metrics").mkdir()

    # Create a small but realistic sample CSV
    sample_data = pd.DataFrame({
        "Unnamed: 0": range(20),
        "Age": [25, 35, 45, 30, 55, 28, 40, 33, 60, 22,
                27, 38, 48, 31, 52, 29, 42, 36, 58, 24],
        "Sex": ["male", "female"] * 10,
        "Job": [2, 1, 2, 3, 1, 2, 1, 2, 3, 1,
                2, 1, 2, 3, 1, 2, 1, 2, 3, 1],
        "Housing": ["own", "rent", "free", "own", "rent"] * 4,
        "Saving accounts": [
            "little", None, "moderate", "rich", "little",
            None, "quite rich", "little", None, "moderate",
            "little", None, "rich", "little", None,
            "moderate", "little", None, "quite rich", "little",
        ],
        "Checking account": [
            None, "little", "moderate", None, "rich",
            "little", None, "moderate", "little", None,
            None, "little", "rich", None, "moderate",
            "little", None, "rich", "little", None,
        ],
        "Credit amount": [
            1000, 5000, 2500, 8000, 1500,
            3000, 7000, 1200, 9000, 800,
            2000, 6000, 1800, 7500, 1100,
            4000, 2200, 8500, 950, 3500,
        ],
        "Duration": [12, 48, 18, 36, 24, 12, 60, 6, 48, 12,
                     18, 36, 12, 48, 6, 24, 18, 60, 12, 36],
        "Purpose": [
            "car", "radio/TV", "education", "furniture/equipment", "car",
            "business", "car", "radio/TV", "furniture/equipment", "car",
            "education", "car", "radio/TV", "business", "car",
            "furniture/equipment", "car", "radio/TV", "education", "car",
        ],
    })
    sample_data.to_csv(tmp_path / "data" / "raw" / "german_credit_data.csv", index=False)

    # Create config.yaml pointing to tmp_path locations
    config = {
        "data": {
            "raw_path": str(tmp_path / "data" / "raw" / "german_credit_data.csv"),
            "processed_path": str(tmp_path / "data" / "processed" / "cleaned.csv"),
            "features_path": str(tmp_path / "data" / "features" / "features.csv"),
            "output_path": str(tmp_path / "data" / "processed" / "labeled_customers.csv"),
            "expected_columns": [
                "Age", "Sex", "Job", "Housing",
                "Saving accounts", "Checking account",
                "Credit amount", "Duration", "Purpose",
            ],
        },
        "model": {
            "type": "clustering",
            "random_state": 42,
            "artifact_path": str(tmp_path / "models") + "/",
            "model_name": "credit_risk_cluster_model",
            "scaler_path": str(tmp_path / "models" / "robust_scaler.pkl"),
        },
    }

    with open(tmp_path / "configs" / "config.yaml", "w") as f:
        yaml.dump(config, f)

    return tmp_path


# ── Ingestion integration tests ───────────────────────────────────────────────

def test_run_ingestion_returns_dataframe(temp_project):
    """run_ingestion should return a DataFrame when config and file are valid."""
    config_path = str(temp_project / "configs" / "config.yaml")
    with patch("src.data.ingestion.load_config") as mock_config:
        with open(config_path) as f:
            mock_config.return_value = yaml.safe_load(f)
        result = run_ingestion()
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 20


def test_run_ingestion_correct_columns(temp_project):
    """run_ingestion should return data with all expected columns."""
    config_path = str(temp_project / "configs" / "config.yaml")
    with patch("src.data.ingestion.load_config") as mock_config:
        with open(config_path) as f:
            mock_config.return_value = yaml.safe_load(f)
        result = run_ingestion()
    expected = ["Age", "Sex", "Job", "Housing", "Saving accounts",
                "Checking account", "Credit amount", "Duration", "Purpose"]
    for col in expected:
        assert col in result.columns


# ── Preprocessing integration tests ──────────────────────────────────────────

def test_run_preprocessing_creates_output_file(temp_project):
    """run_preprocessing should create cleaned.csv in data/processed/."""
    config_path = str(temp_project / "configs" / "config.yaml")
    with patch("src.data.preprocessing.load_config") as mock_config:
        with open(config_path) as f:
            mock_config.return_value = yaml.safe_load(f)
        run_preprocessing()
    output_file = temp_project / "data" / "processed" / "cleaned.csv"
    assert output_file.exists()


def test_run_preprocessing_output_has_no_nulls(temp_project):
    """Cleaned output should have zero null values."""
    config_path = str(temp_project / "configs" / "config.yaml")
    with patch("src.data.preprocessing.load_config") as mock_config:
        with open(config_path) as f:
            mock_config.return_value = yaml.safe_load(f)
        result = run_preprocessing()
    assert result.isnull().sum().sum() == 0


def test_run_preprocessing_output_has_snake_case_columns(temp_project):
    """Cleaned output should use snake_case column names."""
    config_path = str(temp_project / "configs" / "config.yaml")
    with patch("src.data.preprocessing.load_config") as mock_config:
        with open(config_path) as f:
            mock_config.return_value = yaml.safe_load(f)
        result = run_preprocessing()
    assert "credit_amount" in result.columns
    assert "saving_accounts" in result.columns
    assert "Credit amount" not in result.columns


def test_run_preprocessing_removes_unnamed_column(temp_project):
    """Unnamed index column should not appear in cleaned output."""
    config_path = str(temp_project / "configs" / "config.yaml")
    with patch("src.data.preprocessing.load_config") as mock_config:
        with open(config_path) as f:
            mock_config.return_value = yaml.safe_load(f)
        result = run_preprocessing()
    unnamed_cols = [c for c in result.columns if "unnamed" in c.lower()]
    assert len(unnamed_cols) == 0


# ── Feature engineering integration tests ────────────────────────────────────

def test_run_feature_engineering_creates_features_file(temp_project):
    """run_feature_engineering should produce features.csv."""
    config_path = str(temp_project / "configs" / "config.yaml")

    # First run preprocessing to create cleaned.csv
    with patch("src.data.preprocessing.load_config") as mock_config:
        with open(config_path) as f:
            mock_config.return_value = yaml.safe_load(f)
        run_preprocessing()

    # Now run feature engineering
    with patch("src.features.build_features.load_config") as mock_config:
        with open(config_path) as f:
            mock_config.return_value = yaml.safe_load(f)
        run_feature_engineering()

    output_file = temp_project / "data" / "features" / "features.csv"
    assert output_file.exists()


def test_run_feature_engineering_all_numeric(temp_project):
    """features.csv should contain only numeric columns."""
    config_path = str(temp_project / "configs" / "config.yaml")

    with patch("src.data.preprocessing.load_config") as mock_config:
        with open(config_path) as f:
            mock_config.return_value = yaml.safe_load(f)
        run_preprocessing()

    with patch("src.features.build_features.load_config") as mock_config:
        with open(config_path) as f:
            mock_config.return_value = yaml.safe_load(f)
        result = run_feature_engineering()

    non_numeric = result.select_dtypes(exclude=["number"]).columns.tolist()
    assert len(non_numeric) == 0, f"Non-numeric columns found: {non_numeric}"


def test_run_feature_engineering_saves_scaler(temp_project):
    """RobustScaler should be saved as a .pkl file."""
    config_path = str(temp_project / "configs" / "config.yaml")

    with patch("src.data.preprocessing.load_config") as mock_config:
        with open(config_path) as f:
            mock_config.return_value = yaml.safe_load(f)
        run_preprocessing()

    with patch("src.features.build_features.load_config") as mock_config:
        with open(config_path) as f:
            mock_config.return_value = yaml.safe_load(f)
        run_feature_engineering()

    scaler_file = temp_project / "models" / "robust_scaler.pkl"
    assert scaler_file.exists()