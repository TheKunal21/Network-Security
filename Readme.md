<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/AWS_S3-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white" />
</p>

# Network Security — Phishing Website Detection (End-to-End MLOps)

An **end-to-end Machine Learning pipeline** for detecting phishing websites, built with production-grade MLOps practices. The system ingests data from MongoDB, trains & evaluates multiple classifiers via automated hyperparameter tuning, tracks experiments on MLflow/DagsHub, and serves real-time predictions through a FastAPI REST API — all containerized with Docker and backed by AWS S3 artifact storage.

---

## Table of Contents

| # | Section |
|---|---------|
| 1 | [Problem Statement](#-problem-statement) |
| 2 | [Key Features](#-key-features) |
| 3 | [Tech Stack](#-tech-stack) |
| 4 | [High-Level Architecture](#-high-level-architecture) |
| 5 | [Project Structure](#-project-structure) |
| 6 | [ML Pipeline Deep Dive](#-ml-pipeline-deep-dive) |
| 7 | [Dataset](#-dataset) |
| 8 | [Models & Evaluation](#-models--evaluation) |
| 9 | [API Endpoints](#-api-endpoints) |
| 10 | [Getting Started](#-getting-started) |
| 11 | [Docker Deployment](#-docker-deployment) |
| 12 | [Environment Variables](#-environment-variables) |
| 13 | [MLflow Experiment Tracking](#-mlflow-experiment-tracking) |
| 14 | [AWS Cloud Integration](#-aws-cloud-integration) |
| 15 | [Logging & Exception Handling](#-logging--exception-handling) |
| 16 | [Future Enhancements](#-future-enhancements) |

---

## Problem Statement

Phishing attacks remain one of the most prevalent cyber threats, causing billions of dollars in damages annually. Attackers create fraudulent websites that mimic legitimate ones to steal sensitive user information such as credentials, credit card numbers, and personal data.

**Objective:** Build an intelligent classification system that analyzes **30 website-level features** (URL structure, domain properties, page behavior, etc.) and predicts whether a given website is **Phishing (1)** or **Legitimate (0)** — enabling proactive threat detection in real time.

---

## Key Features

| Category | Details |
|----------|---------|
| **Automated ML Pipeline** | End-to-end pipeline covering ingestion → validation → transformation → training → evaluation |
| **Multi-Model Comparison** | 6 classifiers benchmarked with GridSearchCV hyperparameter tuning |
| **Data Drift Detection** | Kolmogorov-Smirnov (KS) test on every feature to flag distribution shifts |
| **Experiment Tracking** | MLflow + DagsHub integration for metric/model versioning |
| **REST API** | FastAPI with train & predict endpoints; CSV upload for batch prediction |
| **Cloud-Native** | Docker containerization, AWS S3 artifact sync |
| **Production Logging** | Timestamped rotating log files with structured formatting |
| **Schema Validation** | YAML-driven schema enforcement for column count & data types |

---

## Tech Stack

```
Category            Technology
─────────────────── ──────────────────────────────
Language            Python 3.10
ML Framework        scikit-learn 1.3.2
Data Handling       Pandas 2.1.4, NumPy 1.26.4
API Framework       FastAPI + Uvicorn
Database            MongoDB Atlas (pymongo)
Experiment Tracking MLflow + DagsHub
Cloud Storage       AWS S3 (via AWS CLI sync)
Containerization    Docker (python:3.10-slim-bullseye)
Serialization       Pickle
Config Management   PyYAML, python-dotenv
Template Engine     Jinja2 (HTML result tables)
```

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        CLIENT / BROWSER                              │
│                  Upload CSV  ─── or ─── Trigger /train               │
└────────────────────────┬─────────────────────┬───────────────────────┘
                         │                     │
                         ▼                     ▼
┌────────────────────────────────────────────────────────────────────┐
│                      FastAPI  (app.py)                              │
│        GET /         GET /train         POST /predict               │
└────────┬───────────────┬──────────────────┬────────────────────────┘
         │               │                  │
         │               ▼                  ▼
         │   ┌──────────────────┐   ┌──────────────────┐
         │   │ Training Pipeline │   │  NetworkModel    │
         │   │   (Orchestrator)  │   │ preprocessor +   │
         │   └──────┬───────────┘   │ model.predict()  │
         │          │               └──────────────────┘
         │          ▼
         │   ┌──────────────────────────────────────────────┐
         │   │           ML PIPELINE STAGES                  │
         │   │                                               │
         │   │  ┌─────────────┐    ┌──────────────────┐     │
         │   │  │  1. DATA     │───▶│  2. DATA          │     │
         │   │  │  INGESTION   │    │  VALIDATION       │     │
         │   │  │  (MongoDB →  │    │  (Schema check,   │     │
         │   │  │   CSV split) │    │   KS drift test)  │     │
         │   │  └─────────────┘    └────────┬─────────┘     │
         │   │                              │               │
         │   │                              ▼               │
         │   │  ┌──────────────────┐   ┌──────────────────┐ │
         │   │  │  4. MODEL        │◀──│  3. DATA          │ │
         │   │  │  TRAINER         │   │  TRANSFORMATION   │ │
         │   │  │  (GridSearchCV,  │   │  (KNN Imputer,    │ │
         │   │  │   best model)    │   │   label encoding) │ │
         │   │  └──────┬───────────┘   └──────────────────┘ │
         │   └─────────┼────────────────────────────────────┘
         │             │
         │             ▼
         │   ┌──────────────────────────────────────────────┐
         │   │           STORAGE & TRACKING                  │
         │   │                                               │
         │   │  ┌──────────┐  ┌────────┐  ┌──────────────┐ │
         │   │  │ Artifacts │  │ MLflow │  │   AWS S3     │ │
         │   │  │ (local)   │  │DagsHub │  │ (sync models │ │
         │   │  │           │  │        │  │ & artifacts) │ │
         │   │  └──────────┘  └────────┘  └──────────────┘ │
         │   └──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│                    MongoDB Atlas                                  │
│              Database: KUNAL_DB                                   │
│              Collection: NetworkData                              │
│              (Raw phishing dataset — 11,055 records)              │
└──────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Network Security Project/
│
├── app.py                          # FastAPI application (train & predict endpoints)
├── main.py                         # Standalone pipeline runner (CLI entrypoint)
├── push_data.py                    # One-time script: CSV → MongoDB ingestion
├── setup.py                        # Package configuration (pip install -e .)
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container image definition
│
├── data_schema/
│   └── schema.yaml                 # Column names, types, numerical columns list
│
├── Network_Data/
│   └── PhisingData.csv             # Raw dataset (30 features + 1 target)
│
├── Networksecurity/                # ← Core Python package
│   ├── __init__.py
│   │
│   ├── Components/                 # Pipeline stage implementations
│   │   ├── data_ingestion.py       #   MongoDB export → CSV → train/test split
│   │   ├── data_validation.py      #   Schema validation + KS drift detection
│   │   ├── data_transformation.py  #   KNN imputation + label encoding
│   │   └── model_trainer.py        #   Multi-model training + GridSearchCV
│   │
│   ├── Constant/
│   │   └── training_pipeline/
│   │       └── __init__.py         # All pipeline constants & hyperparameters
│   │
│   ├── entity/
│   │   ├── config_entity.py        # @dataclass configs for each pipeline stage
│   │   └── artifact_entity.py      # @dataclass artifacts (stage outputs)
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py    # Orchestrates all 4 stages + S3 sync
│   │   └── batch_prediction.py     # Batch inference pipeline
│   │
│   ├── cloud/
│   │   └── s3_syncer.py            # AWS S3 sync (upload/download artifacts)
│   │
│   ├── utils/
│   │   ├── main_utils/
│   │   │   └── utils.py            # YAML I/O, pickle, numpy, GridSearchCV eval
│   │   └── ml_utils/
│   │       ├── metric/
│   │       │   └── classification_metric.py  # F1, Precision, Recall calculator
│   │       └── model/
│   │           └── estimater.py    # NetworkModel wrapper (preprocessor + model)
│   │
│   ├── Exception/
│   │   └── exception.py            # Custom exception with file & line tracking
│   │
│   └── Logging/
│       └── logger.py               # Timestamped file-based logging config
│
├── templates/
│   └── table.html                  # Jinja2 template for prediction results
│
├── final_model/                    # Production-ready model & preprocessor (.pkl)
├── Artifact/                       # Timestamped pipeline run artifacts
├── logs/                           # Application logs (auto-generated)
├── prediction_output/
│   └── output.csv                  # Last batch prediction results
└── valid_data/                     # Validated data snapshots
```

---

## ML Pipeline Deep Dive

### Stage 1 — Data Ingestion

| Aspect | Detail |
|--------|--------|
| **Source** | MongoDB Atlas (`KUNAL_DB.NetworkData`) |
| **Process** | Export collection → DataFrame → drop `_id`/`id` → replace `'na'` with `NaN` → save to feature store CSV |
| **Split** | 80/20 train-test split (`sklearn.model_selection.train_test_split`) |
| **Output Artifact** | `DataIngestionArtifact(trained_file_path, test_file_path)` |

### Stage 2 — Data Validation

| Aspect | Detail |
|--------|--------|
| **Schema Check** | Validates column count against `data_schema/schema.yaml` (31 columns) |
| **Numerical Check** | Ensures all 30 numerical features exist in both train & test sets |
| **Drift Detection** | Kolmogorov-Smirnov two-sample test per feature (threshold: p < 0.05) |
| **Drift Report** | YAML report saved to `Artifact/<timestamp>/data_validation/drift_report/report.yaml` |
| **Output Artifact** | `DataValidationArtifact(validation_status, valid_train/test_paths, drift_report_path)` |

### Stage 3 — Data Transformation

| Aspect | Detail |
|--------|--------|
| **Missing Values** | `KNNImputer(n_neighbors=3, weights='uniform')` via sklearn Pipeline |
| **Label Encoding** | Target column mapped: `-1 → 0` (Legitimate), `1 → 1` (Phishing) |
| **Serialization** | Preprocessor saved as `.pkl`; transformed arrays saved as `.npy` |
| **Output Artifact** | `DataTransformationArtifact(transformed_train/test_paths, preprocessor_path)` |

### Stage 4 — Model Training & Evaluation

| Aspect | Detail |
|--------|--------|
| **Models** | Logistic Regression, KNN, Decision Tree, Random Forest, AdaBoost, Gradient Boosting |
| **Tuning** | `GridSearchCV` with model-specific hyperparameter grids |
| **Metrics** | F1 Score, Precision, Recall (via `ClassificationMetricArtifact`) |
| **Selection** | Best model chosen by highest test score across all candidates |
| **Tracking** | MLflow logs metrics + model artifact → DagsHub remote server |
| **Output Artifact** | `ModelTrainerArtifact(model_path, train_metrics, test_metrics)` |

---

## Dataset

| Property | Value |
|----------|-------|
| **Name** | Phishing Websites Dataset |
| **Records** | ~11,055 |
| **Features** | 30 (all integer-encoded binary/ternary) |
| **Target** | `Result` — `1` (Phishing) / `0` (Legitimate) |
| **Source** | UCI Machine Learning Repository |

### Feature Categories

| Category | Example Features |
|----------|-----------------|
| **URL-based** | `having_IP_Address`, `URL_Length`, `Shortining_Service`, `having_At_Symbol`, `Prefix_Suffix` |
| **Domain-based** | `SSLfinal_State`, `Domain_registeration_length`, `age_of_domain`, `DNSRecord` |
| **Page-based** | `Request_URL`, `URL_of_Anchor`, `Links_in_tags`, `SFH`, `Submitting_to_email` |
| **Behavioral** | `Redirect`, `on_mouseover`, `RightClick`, `popUpWidnow`, `Iframe` |
| **Reputation** | `web_traffic`, `Page_Rank`, `Google_Index`, `Links_pointing_to_page`, `Statistical_report` |

---

## Models & Evaluation

| Model | Hyperparameter Search Space |
|-------|----------------------------|
| **Logistic Regression** | Default parameters |
| **K-Nearest Neighbors** | `n_neighbors`: [3,5,7,9,11], `weights`: [uniform, distance], `algorithm`: [auto, ball_tree, kd_tree, brute] |
| **Decision Tree** | `criterion`: [gini, entropy, log_loss], `splitter`: [best, random], `max_features`: [sqrt, log2] |
| **Random Forest** | `n_estimators`: [8–256], `max_depth`: [None,10,20], `criterion`: [gini, entropy, log_loss] |
| **AdaBoost** | `n_estimators`: [8–256], `learning_rate`: [0.01–1.0] |
| **Gradient Boosting** | `n_estimators`: [8–256], `learning_rate`: [0.01–1.0], `subsample`: [0.6–0.9], `loss`: [log_loss, exponential] |

**Evaluation Metrics:**
- **F1 Score** — Harmonic mean of precision & recall (primary metric)
- **Precision** — Ratio of true phishing detections to all phishing predictions
- **Recall** — Ratio of detected phishing sites to all actual phishing sites

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Redirects to Swagger UI (`/docs`) |
| `GET` | `/train` | Triggers the full training pipeline (ingestion → S3 sync) |
| `POST` | `/predict` | Upload a CSV file → returns HTML table with predictions |

### Predict Endpoint — Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: text/html" \
  -F "file=@test_data.csv"
```

### Predict Endpoint — Response

Returns an HTML page with a styled table showing all input features plus a `predicted_column` (0 = Legitimate, 1 = Phishing). Results are also saved to `prediction_output/output.csv`.

---

## Getting Started

### Prerequisites

- Python 3.10+
- MongoDB Atlas account (or local MongoDB instance)
- AWS CLI configured (for S3 sync — optional)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/TheKunal21/Network-Security.git
cd Network-Security

# 2. Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package in editable mode
pip install -e .

# 5. Set up environment variables (create a .env file)
echo MONGO_DB_URL=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/ > .env
echo DAGSHUB_TOKEN=<your_dagshub_token> >> .env
```

### Load Data into MongoDB (One-Time)

```bash
python push_data.py
```

### Run Training Pipeline

```bash
# Option A: CLI
python main.py

# Option B: API
python app.py
# Then visit: http://localhost:8000/train
```

### Run Prediction Server

```bash
python app.py
# API docs: http://localhost:8000/docs
# Upload CSV via POST /predict
```

---

## Docker Deployment

```bash
# Build the image
docker build -t network-security:latest .

# Run the container
docker run -d \
  -p 8000:8000 \
  -e MONGO_DB_URL="mongodb+srv://..." \
  -e DAGSHUB_TOKEN="..." \
  --name ns-api \
  network-security:latest
```

Access the API at `http://localhost:8000/docs`.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MONGO_DB_URL` | Yes | MongoDB Atlas connection string |
| `DAGSHUB_TOKEN` | No | DagsHub token for MLflow remote tracking |
| `AWS_ACCESS_KEY_ID` | No | AWS credentials for S3 artifact sync |
| `AWS_SECRET_ACCESS_KEY` | No | AWS credentials for S3 artifact sync |

---

## MLflow Experiment Tracking

Experiments are tracked via **MLflow** with **DagsHub** as the remote tracking server.

```
Tracked per run:
├── Metrics
│   ├── f1_score
│   ├── precision
│   └── recall
└── Artifacts
    └── model (sklearn model object)
```

**Dashboard:** [https://dagshub.com/TheKunal21/Network-Security](https://dagshub.com/TheKunal21/Network-Security)

---

## AWS Cloud Integration

| Component | Purpose |
|-----------|---------|
| **S3 Bucket** | Stores pipeline artifacts and final trained models |
| **Sync Direction** | Local → S3 (after every successful training run) |
| **Path Convention** | `s3://<bucket>/artifact/<timestamp>/` and `s3://<bucket>/saved_model/<timestamp>/` |

The `S3Sync` utility class wraps `aws s3 sync` for bidirectional folder synchronization.

---

## Logging & Exception Handling

### Logging

- **Location:** `logs/<YYYY-MM-DD_HH-MM-SS>/<YYYY-MM-DD_HH-MM-SS>.log`
- **Format:** `[timestamp] LEVEL - module - message`
- **Level:** `INFO` (configurable)

### Custom Exception

`NetworkSecurityException` captures:
- Original error message
- Source file name
- Exact line number

This enables precise debugging across all pipeline stages.

---

## Future Enhancements

- [ ] Add CI/CD pipeline (GitHub Actions) for automated training & deployment
- [ ] Implement model monitoring with data drift alerts in production
- [ ] Add deep learning models (e.g., Neural Network classifier) for comparison
- [ ] Deploy on AWS EC2 / ECS with load balancing
- [ ] Add authentication & rate limiting to FastAPI endpoints
- [ ] Implement A/B testing framework for model comparison in production
- [ ] Add Grafana dashboard for real-time prediction monitoring

---

## Author

**Kunal Saini**
- Email: cryptocoffee01@gmail.com
- GitHub: [TheKunal21](https://github.com/TheKunal21)

---

<p align="center">
  <b>If this project helped you, consider giving it a star on GitHub!</b>
</p>