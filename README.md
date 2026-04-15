# Customer Segmentation (RFM + KMeans)

## Project overview
An end-to-end machine learning project that **segments customers** from transaction data using classic **RFM features** and **KMeans clustering**. The project emphasizes a complete, reproducible workflow (data ingestion → feature engineering → model selection → artifact outputs) suitable for an ML portfolio.

- **Goal**: group customers into actionable segments based on purchase behavior (how recently, how often, and how much they buy).
- **Focus**: deliver a clean end-to-end pipeline with clear outputs and explainable segments (not chasing “perfect” clustering).
- **Status**: ✅ **ML pipeline complete** (RFM + KMeans + saved artifacts). API/database folders currently contain **starter scaffolding**.

## What this project does
- **Input**: `data/raw/supermarket_transactions.xlsx`
- **Pipeline**:
  - **Ingest & clean** transactions (validate required columns, parse dates, remove invalid/zero spend rows)
  - **Feature engineering (RFM)** per `customer_id`:
    - **Recency**: days since last purchase (lower is better)
    - **Frequency**: number of transactions
    - **Monetary**: total spend
  - **Preprocessing**: apply `log1p` to skewed features (frequency, monetary) then `StandardScaler`
  - **Modeling**: train KMeans and **select \(k\)** via silhouette score search (\(k=2..8\))
  - **Reporting**: export a per-customer segment table + segment-level profile summary

## Key outputs
Running the pipeline creates the following artifacts:
- **Cleaned transactions**: `data/processed/cleaned_transactions.csv`
- **Trained model**: `models/kmeans.joblib`
- **Scaler + feature columns**: `models/scaler.joblib`
- **Customers with segments**: `models/customers_with_segments.csv`
- **Segment profile summary**: `models/segment_profiles.csv`
- **Training metrics** (chosen \(k\), silhouette scores): `models/training_metrics.json`

## Project structure
```
Customer Segmentation
├── app/                          # FastAPI scaffolding (starter)
│   ├── main.py                   # FastAPI app object (starter)
│   ├── model_handler.py          # placeholder for loading model artifacts
│   ├── schemas.py                # placeholder for request/response schemas
│   └── api_v1/
│       └── endpoints.py          # placeholder for API routes
├── data/
│   ├── raw/
│   │   └── supermarket_transactions.xlsx
│   └── processed/
│       └── cleaned_transactions.csv
├── database/                     # Database scaffolding (starter)
│   ├── db_session.py
│   ├── models.py
│   └── queries.sql
├── ml/
│   ├── transformations.py        # RFM + scaling utilities
│   └── train.py                  # KMeans training + artifact export
├── models/                       # Saved artifacts and reports
│   ├── kmeans.joblib
│   ├── scaler.joblib
│   ├── customers_with_segments.csv
│   ├── segment_profiles.csv
│   └── training_metrics.json
├── notebooks/                    # (optional) experiments / EDA
├── scripts/
│   ├── ingest_data.py            # Excel -> cleaned CSV
│   └── run_pipeline.py           # end-to-end runner
├── tests/                        # unit/integration test folders
├── requirements.txt
└── README.md
```

## Technical stack
- **Python**
- **Data processing**: Pandas, NumPy, SciPy, OpenPyXL
- **Machine learning**: scikit-learn, joblib
- **Visualization (optional)**: Matplotlib, Seaborn, Plotly
- **API scaffolding**: FastAPI, Uvicorn, Pydantic
- **DB scaffolding**: SQLAlchemy, Psycopg

## Getting started
### Prerequisites
- Python 3.9+ recommended
- `pip`

### Install
From the project root:

```bash
pip install -r requirements.txt
```

### Quick start (run the full pipeline)
```bash
python -u scripts/run_pipeline.py
```

## How it works (high level)
- **RFM creation**: per customer, compute last purchase date, count of transactions, and total spend; then derive recency in days.
- **Scaling**: KMeans is distance-based, so we scale features after applying `log1p` to reduce skew.
- **Choosing segments**: try several \(k\) values and pick the best silhouette score (simple and beginner-friendly model selection).

## Notes and assumptions
- **Required columns** in the input Excel: `customer_id`, `timestamp`, `total_amount`
- **Recency reference date**: defaults to \(\max(timestamp) + 1\) day (so the most recent buyer has low recency)
- **Transaction filtering**: removes missing timestamps/amounts and non-positive totals

## Future improvements
- Add **EDA notebook** and visuals for segment interpretation (RFM distributions, segment radar charts)
- Add **API endpoints** to:
  - return a customer’s segment by `customer_id`
  - score a new customer summary (recency/frequency/monetary) into a segment
- Add **database-backed ingestion** (store transactions, compute RFM in SQL, schedule refresh)
- Improve model selection (Elbow method, stability checks, business constraints)
- Add tests for transformations and artifact outputs
