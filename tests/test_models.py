"""
tests/test_models.py

Tests for src/models/train.py and src/models/evaluate.py
Tests model training logic and cluster evaluation logic.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from src.models.evaluate import assign_risk_labels, build_numeric_profiles
from src.models.train import find_optimal_k, train_final_model


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_feature_matrix():
    """Small numeric matrix that simulates scaled features.csv."""
    np.random.seed(42)
    # 60 samples with 5 features — small but enough to cluster
    return np.random.randn(60, 5)


@pytest.fixture
def sample_cleaned_dataframe():
    """Simulates cleaned.csv with original readable values."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "age": np.random.randint(20, 70, n),
        "sex": np.random.choice(["male", "female"], n),
        "job": np.random.randint(0, 4, n),
        "housing": np.random.choice(["own", "rent", "free"], n),
        "saving_accounts": np.random.choice(
            ["none", "little", "moderate", "quite rich", "rich"], n
        ),
        "checking_account": np.random.choice(
            ["none", "little", "moderate", "rich"], n
        ),
        "credit_amount": np.random.randint(500, 15000, n),
        "duration": np.random.randint(4, 72, n),
        "purpose": np.random.choice(["car", "education", "radio/TV"], n),
    })


# ── Training tests ─────────────────────────────────────────────────────────────

def test_train_final_model_returns_kmeans(sample_feature_matrix):
    """train_final_model should return a fitted KMeans object."""
    model = train_final_model(sample_feature_matrix, n_clusters=3, random_state=42)
    assert isinstance(model, KMeans)


def test_train_final_model_correct_cluster_count(sample_feature_matrix):
    """Model should have exactly the number of clusters we asked for."""
    n_clusters = 3
    model = train_final_model(sample_feature_matrix, n_clusters=n_clusters, random_state=42)
    assert model.n_clusters == n_clusters


def test_train_final_model_labels_cover_all_samples(sample_feature_matrix):
    """Every sample should get a cluster label."""
    model = train_final_model(sample_feature_matrix, n_clusters=3, random_state=42)
    assert len(model.labels_) == len(sample_feature_matrix)


def test_find_optimal_k_returns_results_for_each_k(sample_feature_matrix):
    """find_optimal_k should return one result per K value tested."""
    k_range = range(2, 5)
    results = find_optimal_k(sample_feature_matrix, k_range, random_state=42)
    assert len(results) == len(k_range)


def test_find_optimal_k_silhouette_is_between_minus1_and_1(sample_feature_matrix):
    """Silhouette score must always be between -1 and 1."""
    results = find_optimal_k(sample_feature_matrix, range(2, 4), random_state=42)
    for r in results:
        assert -1 <= r["silhouette"] <= 1


def test_find_optimal_k_inertia_decreases_as_k_increases(sample_feature_matrix):
    """Inertia should always decrease as K increases — this is guaranteed."""
    results = find_optimal_k(sample_feature_matrix, range(2, 6), random_state=42)
    inertias = [r["inertia"] for r in results]
    for i in range(len(inertias) - 1):
        assert inertias[i] >= inertias[i + 1], (
            f"Inertia did not decrease: K={i+2} inertia={inertias[i]} "
            f"but K={i+3} inertia={inertias[i+1]}"
        )


# ── Evaluation tests ───────────────────────────────────────────────────────────

def test_assign_risk_labels_returns_three_labels(sample_cleaned_dataframe):
    """assign_risk_labels should return exactly 3 labels for K=3."""
    labels = np.random.randint(0, 3, len(sample_cleaned_dataframe))
    profiles = build_numeric_profiles(sample_cleaned_dataframe, labels)
    risk_map = assign_risk_labels(profiles)
    assert len(risk_map) == 3


def test_assign_risk_labels_uses_correct_label_names(sample_cleaned_dataframe):
    """Labels should be exactly low_risk, medium_risk, high_risk."""
    labels = np.random.randint(0, 3, len(sample_cleaned_dataframe))
    profiles = build_numeric_profiles(sample_cleaned_dataframe, labels)
    risk_map = assign_risk_labels(profiles)
    assert set(risk_map.values()) == {"low_risk", "medium_risk", "high_risk"}


def test_assign_risk_labels_each_cluster_gets_unique_label(sample_cleaned_dataframe):
    """No two clusters should share the same risk label."""
    labels = np.random.randint(0, 3, len(sample_cleaned_dataframe))
    profiles = build_numeric_profiles(sample_cleaned_dataframe, labels)
    risk_map = assign_risk_labels(profiles)
    assert len(set(risk_map.values())) == len(risk_map)


def test_build_numeric_profiles_has_correct_clusters(sample_cleaned_dataframe):
    """Profiles should have one row per cluster."""
    n_clusters = 3
    labels = np.random.randint(0, n_clusters, len(sample_cleaned_dataframe))
    profiles = build_numeric_profiles(sample_cleaned_dataframe, labels)
    assert len(profiles) == n_clusters


def test_build_numeric_profiles_contains_expected_columns(sample_cleaned_dataframe):
    """Profiles must contain the key financial columns used for risk scoring."""
    labels = np.random.randint(0, 3, len(sample_cleaned_dataframe))
    profiles = build_numeric_profiles(sample_cleaned_dataframe, labels)
    for col in ["avg_credit_amount", "avg_duration", "avg_saving", "avg_checking"]:
        assert col in profiles.columns, f"Missing column: {col}"