from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    # When run from project root (e.g., via scripts/run_pipeline.py)
    from ml.transformations import build_rfm_features, build_segment_profiles, make_feature_matrix
except ModuleNotFoundError:
    # When run as `python -u ml/train.py`
    from transformations import build_rfm_features, build_segment_profiles, make_feature_matrix


DEFAULT_PROCESSED_CSV = Path("data/processed/cleaned_transactions.csv")
DEFAULT_MODEL_DIR = Path("models")


def _pick_best_k(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[int, dict]:
    """
    Pick k using silhouette score (simple + beginner friendly).
    """
    results: list[dict] = []
    best = {"k": None, "silhouette": -1.0}

    for k in range(k_min, k_max + 1):
        if verbose:
            print(f"Trying k={k} ...")
        model = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = model.fit_predict(X)
        score = float(silhouette_score(X, labels))
        if verbose:
            print(f"  silhouette={score:.4f}")
        row = {"k": k, "silhouette": score}
        results.append(row)
        if score > best["silhouette"]:
            best = {"k": k, "silhouette": score}

    return int(best["k"]), {"candidates": results, "best": best}


def train_segmentation_model(
    processed_csv: Path = DEFAULT_PROCESSED_CSV,
    model_dir: Path = DEFAULT_MODEL_DIR,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    if not processed_csv.exists():
        raise FileNotFoundError(
            f"Could not find processed data at {processed_csv}. "
            "Run scripts/ingest_data.py first."
        )

    model_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading processed data: {processed_csv}")
    transactions = pd.read_csv(processed_csv)
    if verbose:
        print(f"Loaded {len(transactions):,} rows. Building RFM features...")
    rfm = build_rfm_features(transactions)
    if verbose:
        print(f"Built RFM for {len(rfm):,} customers. Scaling features...")
    X, scaler, feature_cols = make_feature_matrix(rfm)

    if verbose:
        print("Selecting best k with silhouette score...")
    best_k, k_search = _pick_best_k(X, k_min=k_min, k_max=k_max, random_state=random_state, verbose=verbose)
    if verbose:
        print(f"Best k selected: {best_k}. Training final KMeans...")
    kmeans = KMeans(n_clusters=best_k, n_init="auto", random_state=random_state)
    segments = kmeans.fit_predict(X)

    rfm_with_segment = rfm.copy()
    rfm_with_segment["segment"] = segments.astype(int)
    profiles = build_segment_profiles(rfm_with_segment, segment_col="segment")

    # Save artifacts
    if verbose:
        print(f"Saving artifacts to: {model_dir}")
    model_path = model_dir / "kmeans.joblib"
    scaler_path = model_dir / "scaler.joblib"
    customers_path = model_dir / "customers_with_segments.csv"
    profiles_path = model_dir / "segment_profiles.csv"
    metrics_path = model_dir / "training_metrics.json"

    joblib.dump(kmeans, model_path)
    joblib.dump({"scaler": scaler, "feature_cols": feature_cols}, scaler_path)
    rfm_with_segment.to_csv(customers_path, index=False)
    profiles.to_csv(profiles_path, index=False)
    metrics_path.write_text(
        json.dumps(
            {
                "best_k": best_k,
                "k_search": k_search,
                "n_customers": int(rfm_with_segment["customer_id"].nunique())
                if "customer_id" in rfm_with_segment.columns
                else int(len(rfm_with_segment)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "best_k": best_k,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "customers_with_segments": str(customers_path),
        "segment_profiles": str(profiles_path),
        "metrics_path": str(metrics_path),
    }


if __name__ == "__main__":
    out = train_segmentation_model()
    print("Training complete")
    for k, v in out.items():
        print(f"- {k}: {v}")