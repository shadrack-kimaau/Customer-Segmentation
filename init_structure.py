import os
from pathlib import Path

def build_specific_structure():
    root = Path("D:/ML/Customer Segmentation")

    # Define the folders
    folders = [
        "app/api_v1",
        "data/raw",
        "data/processed",
        "data/engine",
        "database",
        "ml",
        "models",
        "notebooks",
        "scripts",
        "tests/unit",
        "tests/integration"
    ]

    # Define files with starter code
    files = {
        "app/__init__.py": "",
        "app/main.py": "# FastAPI Entry\nfrom fastapi import FastAPI\napp = FastAPI()",
        "app/schemas.py": "# Pydantic models",
        "app/model_handler.py": "# Load joblib files",
        "app/api_v1/endpoints.py": "# API routes",
        "database/db_session.py": "# SQL Connection",
        "database/models.py": "# ORM Models",
        "database/queries.sql": "-- SQL for RFM aggregation",
        "ml/train.py": "# Clustering Logic",
        "ml/transformations.py": "# Scaling/RFM Pipelines",
        "scripts/ingest_data.py": "# CSV to SQL",
        "requirements.txt": "fastapi\nuvicorn\npandas\nscikit-learn\nsqlalchemy\npsycopg2-binary",
        ".gitignore": ".venv/\n__pycache__/\n*.csv\n*.joblib\n*.bin\n.env",
        "README.md": f"# Customer Segmentation\nLocation: {root}"
    }

    print(f"--- Initializing Project at: {root} ---")

    # Create directories
    for folder in folders:
        path = root / folder
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {path}")

    # Create files
    for file_path, content in files.items():
        f = root / file_path
        if not f.exists():
            f.write_text(content)
            print(f"Created File: {file_path}")

    print("\n✅ All folders and placeholder files created successfully!")

if __name__ == "__main__":
    build_specific_structure()