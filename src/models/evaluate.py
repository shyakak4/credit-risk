"""
src/models/evaluate.py

Responsibility: Interpret what each cluster means in business terms.
- Analyze cluster characteristics using original readable data
- Assign risk labels (low / medium / high) using financial logic
- Save labeled dataset with risk labels attached to each customer
- Write eval_metrics.json for DVC tracking
- Log everything to MLflow

This is where data science meets business understanding.
The model found 3 groups — we figure out which one is which risk level.
"""

import json
import logging
from pathlib import Path

import joblib
import mlflow
import numpy as np
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


def build_numeric_profiles(
    df_original: pd.DataFrame, labels: np.ndarray
) -> pd.DataFrame:
    """
    Build cluster profiles using numeric-safe versions of the features.

    The processed CSV still has string values like 'little', 'own' because
    preprocessing only cleans — encoding happens in build_features.py.
    For evaluation we re-encode the key columns numerically just for
    the purpose of comparing cluster characteristics.

    We do NOT use the scaled features.csv for this because scaled values
    are hard to interpret (credit_amount of -0.08 means nothing to a human).
    We use original values where possible and encode categoricals for math.
    """
    df = df_original.copy()
    df["cluster"] = labels

    # Re-encode account columns numerically for risk score calculation
    saving_order = {"none": 0, "little": 1, "moderate": 2, "quite rich": 3, "rich": 4}
    checking_order = {"none": 0, "little": 1, "moderate": 2, "rich": 3}

    df["saving_num"] = df["saving_accounts"].map(saving_order)
    df["checking_num"] = df["checking_account"].map(checking_order)

    # Build profile per cluster using interpretable columns
    profiles = (
        df.groupby("cluster")
        .agg(
            avg_credit_amount=("credit_amount", "mean"),
            avg_duration=("duration", "mean"),
            avg_age=("age", "mean"),
            avg_saving=("saving_num", "mean"),
            avg_checking=("checking_num", "mean"),
            count=("age", "count"),
        )
        .round(2)
    )

    logger.info("Cluster profiles (original values):")
    logger.info(f"\n{profiles.to_string()}")

    return profiles


def assign_risk_labels(profiles: pd.DataFrame) -> dict:
    """
    Assign low/medium/high risk labels based on financial logic.

    HIGH RISK profile:
      - High credit amount (bank is exposed to large loss)
      - Long duration (more time = more uncertainty of repayment)
      - Low savings score (customer has no financial cushion)
      - Low checking score (customer has no liquidity)

    We compute a composite risk score per cluster:
      risk = avg_credit_amount + avg_duration - avg_saving - avg_checking

    The cluster with highest score = high_risk
    The cluster with lowest score  = low_risk
    Middle one                     = medium_risk
    """

    # Normalize each metric to 0-1 range so they contribute equally
    def normalize(series):
        range_ = series.max() - series.min()
        if range_ == 0:
            return series * 0
        return (series - series.min()) / range_

    profiles["risk_score"] = (
        normalize(profiles["avg_credit_amount"])
        + normalize(profiles["avg_duration"])
        - normalize(profiles["avg_saving"])
        - normalize(profiles["avg_checking"])
    )

    sorted_clusters = profiles["risk_score"].sort_values()
    risk_labels = ["low_risk", "medium_risk", "high_risk"]

    risk_map = {}
    for i, cluster_id in enumerate(sorted_clusters.index):
        risk_map[cluster_id] = risk_labels[i]
        logger.info(
            f"  Cluster {cluster_id} -> {risk_labels[i]} "
            f"| risk_score={sorted_clusters[cluster_id]:.4f} "
            f"| n={int(profiles.loc[cluster_id, 'count'])} customers"
        )

    return risk_map


def run_evaluation() -> pd.DataFrame: # pragma: no cover
    """
    Main evaluation function. Called by DVC pipeline.
    Reads processed data + trained model, produces labeled output.
    """
    config = load_config()

    processed_path = config["data"]["processed_path"]
    model_path = (
        config["model"]["artifact_path"] + config["model"]["model_name"] + ".pkl"
    )
    output_path = config["data"]["output_path"]

    logger.info("--- Starting evaluation ---")

    # Load everything
    df_original = pd.read_csv(processed_path)
    model = joblib.load(model_path)

    labels = model.labels_
    logger.info(f"Loaded model with {model.n_clusters} clusters")
    logger.info(f"Total customers to label: {len(labels)}")

    # Build interpretable cluster profiles
    profiles = build_numeric_profiles(df_original, labels)

    # Assign risk labels based on financial logic
    logger.info("Assigning risk labels:")
    risk_map = assign_risk_labels(profiles)

    # Attach cluster and risk label to original data
    df_result = df_original.copy()
    df_result["cluster"] = labels
    df_result["risk_label"] = df_result["cluster"].map(risk_map)

    # ── MLflow logging ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credit-risk-clustering")

    with mlflow.start_run(run_name="cluster_evaluation"):
        for cluster_id, risk_label in risk_map.items():
            segment = df_result[df_result["cluster"] == cluster_id]
            mlflow.log_metric(f"{risk_label}_count", len(segment))
            mlflow.log_metric(
                f"{risk_label}_avg_credit", round(segment["credit_amount"].mean(), 2)
            )
            mlflow.log_metric(
                f"{risk_label}_avg_duration", round(segment["duration"].mean(), 2)
            )
            mlflow.log_metric(f"{risk_label}_avg_age", round(segment["age"].mean(), 2))

    # ── Save eval metrics JSON for DVC ─────────────────────────────────────────
    risk_counts = df_result["risk_label"].value_counts().to_dict()
    risk_avg_credit = (
        df_result.groupby("risk_label")["credit_amount"].mean().round(2).to_dict()
    )
    risk_avg_duration = (
        df_result.groupby("risk_label")["duration"].mean().round(2).to_dict()
    )

    eval_metrics = {
        "risk_distribution": risk_counts,
        "avg_credit_amount_by_risk": risk_avg_credit,
        "avg_duration_by_risk": risk_avg_duration,
        "risk_map": {str(k): v for k, v in risk_map.items()},
    }

    Path("metrics").mkdir(exist_ok=True)
    with open("metrics/eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info("Eval metrics saved to metrics/eval_metrics.json")

    # ── Save labeled dataset ───────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(output_path, index=False)
    logger.info(f"Labeled dataset saved to: {output_path}")

    # ── Print human-readable summary ───────────────────────────────────────────
    summary = (
        df_result.groupby("risk_label")
        .agg(
            customers=("cluster", "count"),
            avg_credit_amount=("credit_amount", "mean"),
            avg_duration_months=("duration", "mean"),
            avg_age=("age", "mean"),
        )
        .round(1)
    )

    logger.info("\n" + "=" * 60)
    logger.info("FINAL RISK SEGMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"\n{summary.to_string()}")
    logger.info("=" * 60)
    logger.info("--- Evaluation complete ---")

    return df_result


if __name__ == "__main__": # pragma: no cover
    run_evaluation()
