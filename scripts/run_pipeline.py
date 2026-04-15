from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.train import train_segmentation_model  # noqa: E402
from scripts.ingest_data import ingest_and_clean  # noqa: E402


def run_all() -> dict:
    """
    End-to-end runner:
      1) Excel -> cleaned CSV
      2) RFM -> scaling -> KMeans -> saved artifacts
    """
    ingest_and_clean()
    return train_segmentation_model(
        processed_csv=Path("data/processed/cleaned_transactions.csv"),
        model_dir=Path("models"),
    )


if __name__ == "__main__":
    out = run_all()
    print("Pipeline complete")
    for k, v in out.items():
        print(f"- {k}: {v}")
