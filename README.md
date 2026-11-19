# Wine Classification & Prediction API

**Week 4 — AI Project Deployment & Capstone**  
*Production-ready Machine Learning pipeline and FastAPI inference service for wine cultivar classification.*

---

## 1. Project Overview

This repository houses an end-to-end Machine Learning and DevOps capstone project that classifies wines into three cultivars based on 13 chemical constituents (such as alcohol, malic acid, flavonoids, and proline).

The project adheres to senior-grade software engineering and MLOps standards:
- Modular, object-oriented, and functional architecture
- Data leakage prevention via encapsulated `scikit-learn` pipelines
- Rigorous holdout evaluation with multi-class metrics (Accuracy, Macro F1, Confusion Matrix, ROC-AUC)
- Serialized production model artifacts (`joblib`) and provenance metadata
- Type-safe, validated REST API via FastAPI & Pydantic v2
- Comprehensive automated testing suite (`pytest`, 85 tests passing, 76% code coverage) and code linting (`ruff`)
- Continuous Integration workflow via GitHub Actions (`.github/workflows/ci.yml`)
- Production containerization via Docker & Docker Compose (`python:3.13-slim`, non-root user)

---

## 2. Project Objectives

1. **Reproducible Pipeline:** Deterministic data extraction, scaling, and train/test splitting.
2. **Rigorous Model Selection:** 5-fold cross-validation across Logistic Regression, Random Forest, and Support Vector Classifier (SVC).
3. **Zero Data Leakage:** Ensure scaling parameters and hyperparameter search are fitted exclusively on training splits.
4. **Holdout Evaluation & Serialization:** Evaluate champion pipeline once on unseen holdout test data and serialize full pipeline.
5. **Production Model Serving:** Low-latency FastAPI microservice validating inputs and returning prediction confidences.
6. **Quality Assurance & DevOps:** 100% automated test coverage for core components, continuous linting, and container readiness.

---

## 3. Technology Stack

- **Language:** Python 3.13.9
- **Core Scientific Stack:** `numpy==2.3.5`, `pandas==2.3.3`, `scikit-learn==1.7.2`, `joblib==1.5.2`
- **Web & API Framework:** `fastapi==0.163.0`, `uvicorn==0.54.0`, `pydantic==2.14.3`
- **Testing & Quality:** `pytest==9.2.0`, `httpx==0.28.1`, `ruff==0.17.5`
- **Containerization:** Docker (planned)

---

## 4. Current Implementation Status

| Component | Status | Details |
| :--- | :--- | :--- |
| **Phase 0: Read-Only Audit** | **Completed** | Full system & dependency audit verified. |
| **Phase 1: Environment & Scaffolding** | **Completed** | Isolated `.venv` created, dependencies pinned, Git initialized, foundation tests passing. |
| **Phase 2: Data Ingestion & Preprocessing** | **Completed** | Dataset loader, inspection module, stratified 80/20 train/test split, unfitted StandardScaler pipeline. |
| **Phase 3: Model Training & CV** | **Completed** | Comparative 5-fold CV & GridSearchCV for LR, RF, SVC on training data only. SVC selected champion (Macro F1: 0.9927). |
| **Phase 4: Evaluation & Serialization** | **Completed** | One-time holdout test evaluation (Accuracy: 0.9444, Macro F1: 0.9407, ROC-AUC: 1.0000). Full pipeline serialized (`best_wine_pipeline.joblib`). |
| **Phase 5: FastAPI REST Service** | **Completed** | Singleton `WineModelService`, Pydantic v2 schemas (`WineFeatures`, `PredictionResponse`), `/health`, `/model-info`, `/predict`, `/predict/batch`, live Uvicorn smoke testing passed. |
| **Phase 6: Automated Testing Suite & CI** | **Completed** | 85 automated tests passing, `pytest-cov` reporting 76% coverage, validated GitHub Actions CI workflow (`.github/workflows/ci.yml`). |
| **Phase 7: Containerization & DevOps** | **Completed** | Production `Dockerfile` (`python:3.13-slim`), `.dockerignore`, `docker-compose.yml`, non-root user `appuser`, healthcheck, Compose static validation passed. |
| **Phase 8: Streamlit Frontend Dashboard** | **Completed** | Full-stack interactive Streamlit dashboard (`src/frontend/app.py`), resilient HTTP client (`WineApiClient`), single & CSV batch prediction, model info visualization, 105 automated tests passing. |

---

## 5. Model Evaluation & Performance Summary

### A. Phase 3 5-Fold Cross-Validation (Training Partition — 142 Samples)
| Model | Baseline CV Macro F1 | Tuned CV Macro F1 | Tuned Std | Best Parameters | Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SVC** | **0.9865** | **0.9927** | **0.0147** | `C=10.0`, `kernel='rbf'`, `gamma='scale'` | **1 (Champion)** |
| **Logistic Regression** | 0.9796 | 0.9796 | 0.0273 | `C=1.0`, `solver='lbfgs'` | 2 |
| **Random Forest** | 0.9789 | 0.9789 | 0.0285 | `n_estimators=100`, `max_depth=None`, `min_samples_split=2` | 3 |

### B. Phase 4 Holdout Test Evaluation (Unseen Test Partition — 36 Samples)
| Metric | Holdout Test Score | 5-Fold CV Mean | Generalization Note |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **0.9444** | 0.9862 | 34 of 36 holdout samples correctly classified |
| **Macro F1-Score** | **0.9407** | 0.9927 | Balanced across all 3 cultivars |
| **Macro Precision** | **0.9583** | 0.9865 | High precision across all classes |
| **Macro Recall** | **0.9333** | 0.9878 | Strong sensitivity across all classes |
| **Weighted F1-Score** | **0.9432** | - | Accounts for class frequency weights |
| **ROC-AUC (OVR Macro)** | **1.0000** | - | Multi-class ranking discrimination |

### Holdout Confusion Matrix
```text
                  Predicted class_0   Predicted class_1   Predicted class_2   Total
Actual class_0           12                   0                   0            12
Actual class_1            0                  14                   0            14
Actual class_2            0                   2                   8            10
```

---

## 6. How to Reload and Predict with the Serialized Pipeline

```python
import joblib
import pandas as pd

# Load serialized pipeline artifact
pipeline = joblib.load("models/best_wine_pipeline.joblib")

# Sample inference payload (raw 13 features)
sample = pd.DataFrame([{
    "alcohol": 14.23, "malic_acid": 1.71, "ash": 2.43, "alcalinity_of_ash": 15.6,
    "magnesium": 127.0, "total_phenols": 2.8, "flavanoids": 3.06, "nonflavanoid_phenols": 0.28,
    "proanthocyanins": 2.29, "color_intensity": 5.64, "hue": 1.04,
    "od280/od315_of_diluted_wines": 3.92, "proline": 1065.0
}])

predicted_class = int(pipeline.predict(sample)[0])
probabilities = pipeline.predict_proba(sample)[0].tolist()

print(f"Predicted Class: {predicted_class}")
print(f"Probabilities: {probabilities}")
```

---

## 7. Repository Structure

```text
Week4_AI_Project_Deployment_Capstone/
├── .gitignore                   # Comprehensive Git ignore rules
├── README.md                    # Project documentation and status
├── requirements.txt             # Pinned reproducible dependencies
├── requirements-ui.txt          # Frontend UI dependencies (pins Streamlit)
├── pytest.ini                   # Pytest configuration
├── run.py                       # Central execution runner (status, serve, ui)
│
├── data/
│   ├── raw/
│   │   └── wine.csv             # Raw extracted dataset snapshot
│   └── processed/
│       ├── X_train.csv          # Training features (142 x 13)
│       ├── X_test.csv           # Holdout test features (36 x 13)
│       ├── y_train.csv          # Training target (142)
│       ├── y_test.csv           # Holdout test target (36)
│       └── dataset_metadata.json # Ingestion & split metadata
│
├── models/
│   ├── best_wine_pipeline.joblib # Serialized Pipeline (StandardScaler + SVC)
│   ├── model_metadata.json      # Complete provenance & environment metadata
│   └── evaluation_metrics.json  # Holdout metrics for API serving
│
├── results/
│   ├── training/
│   │   ├── cv_comparison.json   # Detailed CV scores and fold metrics
│   │   ├── cv_comparison.csv    # Tabular baseline vs tuned comparison
│   │   ├── best_model_metadata.json # Champion selection metadata
│   │   └── hyperparameter_search_results.csv # Grid search table
│   └── evaluation/
│       ├── final_metrics.json   # Holdout test metrics
│       ├── classification_report.json # Detailed per-class metrics
│       ├── confusion_matrix.csv # Confusion matrix tabular
│       ├── confusion_matrix.json # Confusion matrix structured
│       ├── test_predictions.csv # Sample-level holdout predictions
│       └── evaluation_summary.md # Comprehensive holdout summary report
│
├── notebooks/.gitkeep           # Exploratory notebooks
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Central paths, constants, and hyperparameters
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py            # Dataset loader and inspection logic
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── pipeline.py          # Data splitting and scaler pipeline factory
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py             # 5-fold CV, tuning & champion selection
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py          # Holdout evaluation & pipeline serialization
│   ├── schemas/                 # Pydantic request/response schemas
│   │   ├── __init__.py
│   │   └── wine_schema.py       # WineFeatures, PredictionResponse, Batch schemas
│   ├── api/                     # FastAPI backend service
│   │   ├── __init__.py
│   │   ├── service.py           # Singleton model loading & inference service
│   │   └── app.py               # Route handlers, lifespan context & error handlers
│   └── frontend/                # Streamlit frontend dashboard
│       ├── __init__.py
│       ├── api_client.py        # Resilient httpx client for FastAPI backend
│       └── app.py               # Multi-tab interactive Streamlit dashboard
│
├── tests/
│   ├── __init__.py
│   ├── test_foundation.py       # Scaffolding & configuration tests (5 tests)
│   ├── test_data_loader.py      # Ingestion & inspection tests (9 tests)
│   ├── test_preprocessing.py    # Preprocessing & leakage tests (10 tests)
│   ├── test_training.py         # Training, CV & model selection tests (13 tests)
│   ├── test_evaluation.py       # Holdout evaluation & metrics tests (9 tests)
│   ├── test_model_serialization.py # Artifact serialization & reload tests (6 tests)
│   ├── test_api.py              # FastAPI endpoints, validation & error tests (24 tests)
│   └── test_frontend.py         # Frontend API client & Streamlit UI tests (21 tests)
│
└── docs/
    └── architecture.md          # Detailed system architecture document
```

---

## 8. Execution & Verification Commands

### Run Full Test Suite with Coverage (106 tests)
```powershell
.\.venv\Scripts\pytest -v --cov=src --cov-report=term-missing
```

### Run Ruff Linter
```powershell
.\.venv\Scripts\ruff check .
```

### Check Dependency Consistency
```powershell
.\.venv\Scripts\python -m pip check
```

### Run Pipeline Stages
```powershell
# 1. Ingest & Inspect
.\.venv\Scripts\python -m src.data.loader

# 2. Stratified Preprocessing
.\.venv\Scripts\python -m src.preprocessing.pipeline

# 3. Model Training & Cross-Validation
.\.venv\Scripts\python -m src.training.train

# 4. Holdout Evaluation & Serialization
.\.venv\Scripts\python -m src.evaluation.evaluate
```

---

## 9. FastAPI REST API Service & Endpoints

### Launch API Server
```powershell
# Option A: Central project CLI runner
.\.venv\Scripts\python run.py serve --port 8000

# Option B: Direct Uvicorn development server
.\.venv\Scripts\uvicorn src.api.app:app --reload --port 8000
```
Interactive API documentation is automatically accessible at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### API Endpoint Reference & Examples

#### 1. Service Health & Readiness Probe
- **Endpoint:** `GET /health`
- **PowerShell Invocation:**
  ```powershell
  Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
  ```
- **Response (200 OK):**
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "model_ready": true,
    "model_name": "Support Vector Classifier (SVC)",
    "version": "1.0.0"
  }
  ```

#### 2. Model Metadata & Performance Probe
- **Endpoint:** `GET /model-info`
- **PowerShell Invocation:**
  ```powershell
  Invoke-RestMethod -Uri "http://127.0.0.1:8000/model-info" -Method Get
  ```
- **Response (200 OK):** Returns pipeline steps (`StandardScaler`, `SVC`), 13 ordered feature names, 3 target cultivars, hyperparameters, Phase 3 CV Macro F1 (`0.9927`), and Phase 4 holdout test metrics (`Accuracy: 0.9444`, `Macro F1: 0.9407`).

#### 3. Single Wine Sample Inference
- **Endpoint:** `POST /predict`
- **PowerShell Invocation:**
  ```powershell
  $body = @{
      alcohol = 13.2
      malic_acid = 1.78
      ash = 2.14
      alcalinity_of_ash = 11.2
      magnesium = 100.0
      total_phenols = 2.65
      flavanoids = 2.76
      nonflavanoid_phenols = 0.26
      proanthocyanins = 1.28
      color_intensity = 4.38
      hue = 1.05
      "od280/od315_of_diluted_wines" = 3.40
      proline = 1050.0
  } | ConvertTo-Json

  Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Body $body -ContentType "application/json"
  ```
- **Response (200 OK):**
  ```json
  {
    "predicted_class": 0,
    "predicted_class_name": "class_0",
    "probabilities": {
      "class_0": 0.9698978654539568,
      "class_1": 0.020295971806231945,
      "class_2": 0.00980616273981104
    },
    "model_name": "Support Vector Classifier (SVC)",
    "model_version": "1.0.0"
  }
  ```

#### 4. Bulk Batch Prediction
- **Endpoint:** `POST /predict/batch`
- **PowerShell Invocation:**
  ```powershell
  $batchPayload = @{
      samples = @($sample1, $sample2)
  } | ConvertTo-Json -Depth 5

  Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict/batch" -Method Post -Body $batchPayload -ContentType "application/json"
  ```
- **Response (200 OK):** Returns sequence-preserved predictions with sample indexes (`0`, `1`, ...), total record counts, and individual probability distributions.

#### 5. Validation Guarantees & Error Codes
- **HTTP 422 Unprocessable Content:** Triggered when any of the 13 required features are missing, unallowed extra fields are included, types are non-numeric, or values are booleans/NaN/Infinity.
- **HTTP 503 Service Unavailable:** Cleanly returned if the model artifact is unloaded or damaged.
- **HTTP 500 Internal Server Error:** Sanitized generic response that shields server internals and prevents stack trace leakage.

---

## 10. Continuous Integration (CI) Workflow

The project includes an automated GitHub Actions CI workflow configured at [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

- **Triggers:** Automatically runs on `push` and `pull_request` targeting `main` or `master` branches.
- **Environment:** Runs on `ubuntu-latest` using `python-version: "3.13"` with pip caching enabled.
- **Pipeline Gates:**
  1. **Checkout:** Clones codebase via `actions/checkout@v4`.
  2. **Python Setup:** Configures Python 3.13 runtime with pip cache via `actions/setup-python@v5`.
  3. **Dependency Installation:** Installs pinned dependencies from `requirements.txt`.
  4. **Dependency Consistency:** Executes `python -m pip check` ensuring zero package conflicts.
  5. **Static Analysis & Linting:** Runs `ruff check .` with 0 permitted errors or formatting discrepancies.
  6. **Automated Testing & Coverage:** Executes all 85 pytest tests and prints terminal code coverage reporting.
- **Security:** Minimal permissions model (`permissions: contents: read`), zero secrets/tokens exposed, no Docker containers or deployment jobs.

---

## 11. Docker Containerization & Local Deployment

The application includes production containerization assets for standalone or orchestrated execution:

### Container Architecture & Security
- **Base Image:** `python:3.13-slim` (official lightweight Debian-based Python image).
- **Security:** Runs under unprivileged user `appuser` (UID/GID `10001`). No root execution.
- **Port:** Exposes documented port `8000`.
- **Health Check:** Regularly probes `http://127.0.0.1:8000/health` using standard Python `urllib.request`. Returns unhealthy if model is unloaded or service is degraded.
- **Configuration:** Environment variables parameterized via `.env.example`.

### Build & Run Commands

```powershell
# 1. Build the Docker image
docker build -t wine-classification-api:latest .

# 2. Run container as daemon
docker run -d `
  --name wine_classification_api `
  -p 8000:8000 `
  --restart unless-stopped `
  wine-classification-api:latest

# 3. View container logs
docker logs -f wine_classification_api

# 4. Stop and remove container
docker stop wine_classification_api
docker rm wine_classification_api
```

### Docker Compose Orchestration

```powershell
# Start service in background
docker compose up -d

# Check service and health status
docker compose ps

# View service logs
docker compose logs -f

# Shut down service cleanly
docker compose down
```

### Host Daemon Requirement Note
Executing `docker build` or `docker compose up` requires an active Docker daemon (e.g. Docker Desktop running on Windows). Static Compose syntax has been verified via `docker compose config`.

---

## 12. Streamlit Frontend Dashboard & User Guide

Phase 8 provides an interactive web dashboard for real-time inference, batch CSV analysis, and model inspection.

### Architecture Decoupling & Independent Processes
The frontend is completely decoupled from the machine learning inference engine and model binaries:
- **FastAPI Backend (Process 1):** Loads `models/best_wine_pipeline.joblib`, validates inputs via Pydantic v2 schemas, runs scikit-learn pipeline inference, and exposes REST endpoints on port `8000`.
- **Streamlit Frontend (Process 2):** Runs an interactive UI on port `8501`, connecting exclusively via HTTP using `WineApiClient` (`httpx`). The frontend **never** loads model weights or imports scikit-learn.

```text
[User Browser] ---> http://localhost:8501 [Streamlit UI (src/frontend/app.py)]
                                                       |
                                               HTTP (httpx Client)
                                                       v
                                     http://localhost:8000 [FastAPI Backend (src/api/app.py)]
                                                       |
                                            [best_wine_pipeline.joblib]
```

### Installation & Startup

#### 1. Install Frontend Dependencies
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
```

#### 2. Launch FastAPI Backend (Terminal 1)
```powershell
# Using central project runner
.\.venv\Scripts\python run.py serve --port 8000

# Or directly via Uvicorn
.\.venv\Scripts\uvicorn src.api.app:app --reload --port 8000
```

#### 3. Launch Streamlit Frontend (Terminal 2)
```powershell
# Using central project runner
.\.venv\Scripts\python run.py ui --port 8501

# Or directly via Streamlit CLI
.\.venv\Scripts\streamlit run src/frontend/app.py --server.port 8501
```

Access the interactive dashboard in your browser at `http://localhost:8501`.

### API URL Configuration
The frontend automatically connects to `http://127.0.0.1:8000` by default. You can configure a custom backend address in two ways:
1. **Environment Variable:** Set `API_BASE_URL` (or `API_URL`) before launching Streamlit:
   ```powershell
   $env:API_BASE_URL = "http://127.0.0.1:8000"
   ```
2. **Dashboard Sidebar:** Modify the **FastAPI Service URL** text box in the sidebar and click **Probe Backend Health**.

### Dashboard Modules & Features

#### A. Single Wine Prediction
- **13 Chemical Constituents:** Enter exact measurements (`alcohol`, `malic_acid`, `ash`, `alcalinity_of_ash`, `magnesium`, `total_phenols`, `flavanoids`, `nonflavanoid_phenols`, `proanthocyanins`, `color_intensity`, `hue`, `od280/od315_of_diluted_wines`, `proline`).
- **Archetype Presets:** Quick-fill the form with typical constituent profiles for Cultivar 0, Cultivar 1, or Cultivar 2.
- **Inference Results:** Displays the predicted cultivar name (`class_0`, `class_1`, or `class_2`), model version, estimated class probabilities formatted as percentages, and an interactive probability bar chart. Includes an explicit disclaimer that these represent model-calculated class probability estimates and are not guaranteed to be empirically calibrated.

#### B. Batch CSV Prediction
- **Sample Template Download:** Download a pre-formatted `wine_batch_sample_template.csv` directly from the dashboard.
- **Pre-flight Validation:** Upload any CSV file (1 to 100 rows). The app automatically validates that all 13 columns are present, checks for extraneous columns, and flags any non-numeric or missing (`NaN`) values with specific row indices before submission.
- **Atomic Inference:** Submits validated records to `POST /predict/batch`.
- **Downloadable Results:** View the predictions table and download `wine_classification_results.csv` containing original inputs alongside predicted classes and estimated class probabilities.

#### C. Model Architecture & Provenance Metrics
- **Live Metadata:** Fetches architecture, scikit-learn version, and hyperparameters directly from `GET /model-info`.
- **Metrics Comparison:** Compares Phase 3 5-fold cross-validation metrics (Mean Macro F1: `0.9927 ± 0.0147`) against Phase 4 holdout evaluation metrics (Accuracy: `0.9444`, Macro F1: `0.9407`, ROC-AUC: `1.0000`).

### Troubleshooting Backend Connection Errors
- **"🔴 Backend Offline" Indicator:** If the sidebar displays "Backend Offline", verify that the FastAPI backend process is running on the expected port (default `8000`).
- **Connection Refused:** Ensure no firewall blocks port `8000` and confirm the server is started with `python run.py serve`.
- **Validation Errors (HTTP 422):** The frontend displays structured field-level error messages returned by Pydantic v2 if any required constituent is out of bounds, non-numeric, or missing.

---

## 13. Docker Compose — Full Application

Phase 8 Slice 2 enables full-stack multi-container orchestration for the FastAPI backend (`wine-api`) and Streamlit frontend (`wine-ui`) via Docker Compose.

### Architecture Overview

```text
Host Browser
     │
     ├──> http://localhost:8501  ──> [wine-ui (Streamlit)]
     │                                      │
     │                                      │ internal network (http://wine-api:8000)
     │                                      v
     └──> http://localhost:8000  ──> [wine-api (FastAPI + Model Artifact)]
```

### Prerequisites
- Docker Desktop installed and running.
- WSL2 backend enabled on Windows (or native Docker engine on Linux/macOS).
- Host ports `8000` and `8501` available (or remapped in `.env`).

### Build & Start Both Services
Run the following single command from the project root:
```powershell
docker compose up --build -d
```
This builds both `wine-api` (using `Dockerfile`) and `wine-ui` (using `Dockerfile.ui`), creates the isolated default network, starts `wine-api`, verifies its health via `http://127.0.0.1:8000/health`, and then starts `wine-ui` once `wine-api` reports healthy.

### Verification & Monitoring
```powershell
# Verify container and health statuses
docker compose ps

# View live logs for both services
docker compose logs -f

# View logs for backend only
docker compose logs -f wine-api

# View logs for frontend only
docker compose logs -f wine-ui
```

### Access URLs
- **Streamlit Web UI:** [http://localhost:8501](http://localhost:8501)
- **FastAPI Interactive Docs (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **FastAPI Alternative Docs (ReDoc):** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **FastAPI Health Probe:** [http://localhost:8000/health](http://localhost:8000/health)
- **FastAPI Model Information:** [http://localhost:8000/model-info](http://localhost:8000/model-info)

### Stopping the Stack
```powershell
# Stop services while preserving container state
docker compose stop

# Restart stopped services
docker compose start

# Teardown containers and network cleanly
docker compose down
```

> [!CAUTION]
> **Data Loss Warning:** Never run `docker compose down -v`. The `-v` flag deletes named volumes and could cause irrecoverable data loss across other local databases or shared containers.

### Configuration (`.env`)
Both services are fully parameterized through environment variables defined in `.env` (copy from `.env.example`):
- `API_PORT`: Host port bound to the FastAPI backend (default: `8000`).
- `UI_PORT`: Host port bound to the Streamlit frontend (default: `8501`).
- `API_BASE_URL`: Backend URL resolution:
  - **Inside Docker Compose:** Set to `http://wine-api:8000` (uses Docker's internal DNS resolution).
  - **Local Development outside Docker:** Set to `http://127.0.0.1:8000`.
- `HOST`, `PORT`, `WORKERS`, `LOG_LEVEL`: Backend Uvicorn server runtime parameters.

### Troubleshooting
- **Backend Not Healthy / Frontend Waiting:** The frontend container waits for the backend to become healthy (`condition: service_healthy`). If the UI is delayed starting, inspect backend logs with `docker compose logs -f wine-api`. Ensure model artifact `models/best_wine_pipeline.joblib` is present and readable.
- **Port Conflicts on 8000 or 8501:** If host ports `8000` or `8501` are already in use by other processes, adjust `API_PORT` or `UI_PORT` in `.env` (e.g. `API_PORT=8080`, `UI_PORT=8502`), then re-run `docker compose up -d`.
- **Streamlit Cannot Reach Backend:** Verify that `wine-ui` has `API_BASE_URL=http://wine-api:8000` set. You can also override the API URL dynamically inside the running Streamlit sidebar settings and click "Probe Backend Health".





