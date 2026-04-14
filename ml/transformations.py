from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


REQUIRED_TRANSACTION_COLUMNS: tuple[str, ...] = ("customer_id", "timestamp", "total_amount")


@dataclass(frozen=True)
class RFMConfig:
    customer_id_col: str = "customer_id"
    timestamp_col: str = "timestamp"
    amount_col: str = "total_amount"
    # If None, we use (max(timestamp) + 1 day) as the reference date
    reference_date: str | pd.Timestamp | None = None


def validate_transactions_df(df: pd.DataFrame, required_cols: Iterable[str] = REQUIRED_TRANSACTION_COLUMNS) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Transactions file is missing required columns: "
            f"{missing}. Found columns: {list(df.columns)}"
        )


def build_rfm_features(transactions: pd.DataFrame, cfg: RFMConfig = RFMConfig()) -> pd.DataFrame:
    """
    Build a simple RFM table:
      - recency_days: days since last purchase (lower is better)
      - frequency: number of transactions
      - monetary: total spend
    """
    validate_transactions_df(
        transactions, required_cols=(cfg.customer_id_col, cfg.timestamp_col, cfg.amount_col)
    )

    df = transactions[[cfg.customer_id_col, cfg.timestamp_col, cfg.amount_col]].copy()
    df[cfg.timestamp_col] = pd.to_datetime(df[cfg.timestamp_col], errors="coerce")
    df[cfg.amount_col] = pd.to_numeric(df[cfg.amount_col], errors="coerce")
    df = df.dropna(subset=[cfg.customer_id_col, cfg.timestamp_col, cfg.amount_col])
    df = df[df[cfg.amount_col] > 0]

    if cfg.reference_date is None:
        reference_date = df[cfg.timestamp_col].max() + pd.Timedelta(days=1)
    else:
        reference_date = pd.to_datetime(cfg.reference_date)

    grouped = df.groupby(cfg.customer_id_col, as_index=False).agg(
        last_purchase=(cfg.timestamp_col, "max"),
        frequency=(cfg.timestamp_col, "count"),
        monetary=(cfg.amount_col, "sum"),
    )
    grouped["recency_days"] = (reference_date - grouped["last_purchase"]).dt.days.astype(int)

    # Keep only features + identifier
    rfm = grouped[[cfg.customer_id_col, "recency_days", "frequency", "monetary"]].copy()
    rfm = rfm.sort_values(cfg.customer_id_col).reset_index(drop=True)
    return rfm


def make_feature_matrix(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """
    Convert RFM table into a numeric feature matrix suitable for clustering.

    We use log1p on skewed columns (frequency, monetary) then StandardScaler.
    """
    feature_cols = ["recency_days", "frequency", "monetary"]
    missing = [c for c in feature_cols if c not in rfm.columns]
    if missing:
        raise ValueError(f"RFM table missing columns {missing}. Found: {list(rfm.columns)}")

    X = rfm[feature_cols].astype(float).copy()
    X["frequency"] = np.log1p(X["frequency"])
    X["monetary"] = np.log1p(X["monetary"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    return X_scaled, scaler, feature_cols


def build_segment_profiles(rfm_with_segment: pd.DataFrame, segment_col: str = "segment") -> pd.DataFrame:
    """
    Aggregate mean/median stats per segment for a simple report.
    """
    needed = {"recency_days", "frequency", "monetary", segment_col}
    missing = [c for c in needed if c not in rfm_with_segment.columns]
    if missing:
        raise ValueError(f"Missing columns for profiling: {missing}")

    prof = (
        rfm_with_segment.groupby(segment_col)
        .agg(
            customers=("customer_id", "nunique") if "customer_id" in rfm_with_segment.columns else (segment_col, "size"),
            recency_days_mean=("recency_days", "mean"),
            recency_days_median=("recency_days", "median"),
            frequency_mean=("frequency", "mean"),
            frequency_median=("frequency", "median"),
            monetary_mean=("monetary", "mean"),
            monetary_median=("monetary", "median"),
        )
        .reset_index()
        .sort_values(segment_col)
    )
    return prof