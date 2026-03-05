"""
src/models/train.py

Responsibility: Train K-Means clustering model on engineered features.
- Find optimal K using Elbow method and Silhouette score
- Train final model with K from params.yaml
- Save model artifact
- Log everything to MLflow

MLOps principle: Every experiment is tracked. Nothing is lost.
You can always go back and see exactly what produced any model.
"""

import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def find_optimal_k(
    X: np.ndarray,
    k_range: range,
    random_state: int,
) -> dict:
    """
    Run K-Means for a range of K values.
    Record inertia (elbow method) and silhouette score for each K.

    Inertia: measures how tight clusters are (lower = better)
    Silhouette: measures separation between clusters (-1 to 1, higher = better)

    We use BOTH because elbow alone can be ambiguous.
    """
    results = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertia = km.inertia_
        sil = silhouette_score(X, labels) if k > 1 else 0.0

        results.append(
            {
                "k": k,
                "inertia": inertia,
                "silhouette": sil,
            }
        )
        logger.info(f"  K={k} | inertia={inertia:.1f} | silhouette={sil:.4f}")

    return results


def train_final_model(
    X: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> KMeans:
    """
    Train the final K-Means model with the chosen K.
    n_init=10 means we run 10 different random initializations
    and keep the best result — prevents getting stuck in bad local minima.
    """
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    model.fit(X)
    logger.info(f"Model trained | K={n_clusters} | inertia={model.inertia_:.1f}")
    return model


def run_training() -> KMeans: # pragma: no cover
    """
    Main training function. Called by DVC pipeline.
    Reads features, trains model, logs to MLflow, saves artifact.
    """
    config = load_config()
    params = load_params()

    features_path = config["data"]["features_path"]
    model_path = (
        config["model"]["artifact_path"] + config["model"]["model_name"] + ".pkl"
    )
    random_state = config["model"]["random_state"]
    n_clusters = params["model"]["n_clusters"]
    k_min = params["model"]["k_min"]
    k_max = params["model"]["k_max"]

    logger.info("─── Starting model training ───")

    # Load features
    df = pd.read_csv(features_path)
    X = df.values
    logger.info(f"Feature matrix: {X.shape}")

    # MLflow experiment
    mlflow.set_experiment("credit-risk-clustering")

    with mlflow.start_run(run_name=f"kmeans_k{n_clusters}"):

        # Log parameters
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_samples", X.shape[0])

        # Find optimal K (logged for visibility in MLflow)
        logger.info(f"Running elbow analysis for K={k_min} to K={k_max}...")
        k_results = find_optimal_k(X, range(k_min, k_max + 1), random_state)

        for result in k_results:
            mlflow.log_metric(f"inertia_k{result['k']}", result["inertia"])
            mlflow.log_metric(f"silhouette_k{result['k']}", result["silhouette"])

        # Train final model
        model = train_final_model(X, n_clusters, random_state)
        labels = model.labels_

        # Final metrics
        final_silhouette = silhouette_score(X, labels)
        mlflow.log_metric("final_silhouette", final_silhouette)
        mlflow.log_metric("final_inertia", model.inertia_)

        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            pct = count / len(labels) * 100
            mlflow.log_metric(f"cluster_{cluster_id}_size", count)
            logger.info(f"  Cluster {cluster_id}: {count} customers ({pct:.1f}%)")

        metrics = {
            "silhouette_score": round(final_silhouette, 4),
            "inertia": round(model.inertia_, 2),
            "n_clusters": n_clusters,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "cluster_sizes": {
                f"cluster_{int(k)}": int(v) for k, v in zip(unique, counts)
            },
        }
        Path("metrics").mkdir(exist_ok=True)
        with open("metrics/train_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved to metrics/train_metrics.json")

        # Save model
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "kmeans_model")

        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Silhouette score: {final_silhouette:.4f}")

    logger.info("─── Training complete ───")
    return model


if __name__ == "__main__": # pragma: no cover
    run_training()
