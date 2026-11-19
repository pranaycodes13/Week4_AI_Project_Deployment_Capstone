# Architecture Specification: Wine Classification & Prediction API

**Project:** Week 4 — AI Project Deployment & Capstone  
**Author:** Senior Machine Learning Engineer & DevOps Architect  
**Current Status:** Phase 5 Complete (FastAPI Backend Service & Input Validation Verified)

---

## 1. System Overview

The Wine Classification & Prediction service is an end-to-end, reproducible machine learning system designed to ingest chemical analysis data of Italian wines and classify samples into one of three wine cultivars. The system bridges machine learning engineering and production DevOps through structured pipelines, rigorous testing, serialized model artifacts, and a containerized FastAPI REST interface.

```
+-------------------------------------------------------------------------------+
|                               System Flowchart                                |
+-------------------------------------------------------------------------------+

 [Scikit-Learn Wine Dataset]
              |
              v (Phase 2: Data Ingestion [Implemented])
     [data/raw/wine.csv]
              |
              v (Phase 2: Stratified Split 80/20 [Implemented])
     +--------+--------+
     |                 |
     v                 v
 [Train Split]    [Holdout Test Split]
  (142 samples)     (36 samples - Unseen)
     |                 |
     v                 |
 [StandardScaler]      |
     | (Fitted on Train)
     v                 |
 [Model Training & CV] | (Phase 3: Logistic Regression, Random Forest, SVC [Implemented])
     | (Champion: SVC) |
     v                 |
 [Fitted Pipeline]     | (StandardScaler + SVC[C=10.0, rbf, gamma=scale])
     |                 |
     +--------+--------+
              |
              v (Phase 4: Holdout Evaluation [Implemented])
     [Holdout Evaluation Metrics] (Accuracy: 0.9444, Macro F1: 0.9407, ROC-AUC: 1.0)
              |
              v (Phase 4: Joblib Serialization [Implemented])
     [models/best_wine_pipeline.joblib] (10.9 KB serialized Pipeline artifact)
              |
              v (Phase 5: FastAPI Service Injection [Implemented])
    +------------------------------------------+
    |           FastAPI REST API               |
    |  - GET  /health (Readiness & Metadata)   |
    |  - GET  /model-info (Features & Metrics) |
    |  - POST /predict (Validated Inference)   |
    |  - POST /predict/batch (Bulk Inference)  |
    +------------------------------------------+
              |
              v (Phase 6 & 7: Pytest & Containerization [Planned])
       [Docker Container / CI Testing]
```

---

## 2. Dataset & Data Ingestion (Implemented — Phase 2)

- **Source:** Scikit-learn Wine recognition dataset (`sklearn.datasets.load_wine`).
- **Samples & Features:** Exactly 178 samples across 13 continuous chemical features:
  `alcohol`, `malic_acid`, `ash`, `alcalinity_of_ash`, `magnesium`, `total_phenols`, `flavanoids`, `nonflavanoid_phenols`, `proanthocyanins`, `color_intensity`, `hue`, `od280/od315_of_diluted_wines`, `proline`.
- **Target Variable:** 3 cultivar classes (`0`, `1`, `2`) representing origin wine varieties.
- **Data Integrity & Inspection:**
  - Null / Missing values: 0 across all features and targets.
  - Duplicate sample rows: 0 duplicates.
  - Target class distribution: Class 0: 59 (33.1%), Class 1: 71 (39.9%), Class 2: 48 (27.0%).
  - Raw snapshot saved to `data/raw/wine.csv` (178 rows × 14 columns including target).

---

## 3. Stratified Partitioning & Preprocessing Design (Implemented — Phase 2)

- **Train/Test Splitting (`src/preprocessing/pipeline.py`):**
  - Partition ratio: 80% Train (142 samples) / 20% Holdout Test (36 samples).
  - Deterministic seed: `random_state=42`.
  - Stratification strategy: `stratify=y` ensures identical class ratios across both partitions:
    - Train (142): Class 0 = 47 (33.1%), Class 1 = 57 (40.1%), Class 2 = 38 (26.8%)
    - Test (36): Class 0 = 12 (33.3%), Class 1 = 14 (38.9%), Class 2 = 10 (27.8%)
- **Data Leakage Safeguards:**
  - Index disjointness: Set intersection of train and test sample indices is strictly empty ($\emptyset$).
  - Separation of transformations: Feature scaling parameters ($\mu, \sigma$) are **never** calculated using the full dataset or the test set.
  - Pipeline factory (`create_preprocessing_pipeline()`): Produces an unfitted `sklearn.pipeline.Pipeline` holding an uncalibrated `StandardScaler`.
  - Fitting rules: The scaler is fitted strictly on `X_train` during training. Holdout test data (`X_test`) is exclusively transformed via the fitted pipeline.

---

## 4. Model Architecture & Selection Strategy (Implemented — Phase 3)

### Candidate Architectures
Every candidate model is packaged inside an `sklearn.pipeline.Pipeline` with `StandardScaler`:
1. **Multinomial Logistic Regression:** Linear classifier with L2 penalty (`max_iter=1000`, `random_state=42`).
2. **Random Forest Classifier:** Non-linear decision tree ensemble (`random_state=42`).
3. **Support Vector Classifier (SVC):** Non-linear kernel classifier with probability calibration (`probability=True`, `random_state=42`).

### Cross-Validation Methodology & Champion Selection
- **Strategy:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` applied exclusively to `X_train` and `y_train`.
- **Tuned Results:**
  - **SVC (Champion):** Mean CV Macro F1 = **0.9927** ($\pm 0.0147$), best params: `C=10.0`, `kernel='rbf'`, `gamma='scale'`.
  - **Logistic Regression:** Mean CV Macro F1 = 0.9796 ($\pm 0.0273$).
  - **Random Forest:** Mean CV Macro F1 = 0.9789 ($\pm 0.0285$).

---

## 5. Holdout Evaluation & Model Serialization (Implemented — Phase 4)

### One-Time Holdout Evaluation
The champion SVC pipeline was fitted on all 142 training samples and evaluated on the unseen 36 holdout test samples:

| Metric | Holdout Test Score | 5-Fold CV Mean | Interpretation |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **0.9444** | 0.9862 | 34 of 36 holdout samples correctly classified |
| **Macro F1-Score** | **0.9407** | 0.9927 | Balanced multi-class performance |
| **Macro Precision** | **0.9583** | 0.9865 | High precision across all classes |
| **Macro Recall** | **0.9333** | 0.9878 | Strong sensitivity across all classes |
| **Weighted F1-Score**| **0.9432** | - | Sample-weighted multi-class F1 |
| **ROC-AUC (OVR Macro)**| **1.0000** | - | Multi-class ranking discrimination |

### Holdout Confusion Matrix
| Actual \\ Predicted | Predicted class_0 | Predicted class_1 | Predicted class_2 | Total | Class Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Actual class_0** | **12** | 0 | 0 | 12 | 100.0% |
| **Actual class_1** | 0 | **14** | 0 | 14 | 100.0% |
| **Actual class_2** | 0 | 2 | **8** | 10 | 80.0% |

> [!NOTE]
> - **CV vs. Holdout Score Difference:** 5-fold CV averaged 0.9927 on 142 training samples, while the holdout test set scored 0.9407. In a small holdout set of 36 samples, each misclassification represents $\approx 2.78\%$ accuracy drop. The 2 misclassifications were class 2 wines predicted as class 1, which lie near the non-linear boundary in the chemical feature space.
> - **ROC-AUC Limitations:** ROC-AUC of 1.0000 indicates ranking discrimination. It does not prove probability calibration or zero error across unseen external distributions.

### Serialized Pipeline Structure (`models/best_wine_pipeline.joblib`)
- **Encapsulated Steps:**
  1. `StandardScaler`: Fitted mean and standard deviation for all 13 features.
  2. `SVC`: Fitted support vectors and Platt probability calibration parameters.
- **Inference Guarantee:** Raw inputs pass directly through `pipeline.predict(X)` or `pipeline.predict_proba(X)` without requiring external scaling scripts.

---

## 6. How to Reload and Use the Serialized Model

```python
import joblib
import pandas as pd
from pathlib import Path

# Load serialized pipeline
model_path = Path("models/best_wine_pipeline.joblib")
pipeline = joblib.load(model_path)

# Predict raw sample (13 chemical features in order)
sample_df = pd.DataFrame([{
    "alcohol": 13.2,
    "malic_acid": 1.78,
    "ash": 2.14,
    "alcalinity_of_ash": 11.2,
    "magnesium": 100.0,
    "total_phenols": 2.65,
    "flavanoids": 2.76,
    "nonflavanoid_phenols": 0.26,
    "proanthocyanins": 1.28,
    "color_intensity": 4.38,
    "hue": 1.05,
    "od280/od315_of_diluted_wines": 3.4,
    "proline": 1050.0
}])

prediction = pipeline.predict(sample_df)[0]
probabilities = pipeline.predict_proba(sample_df)[0]
print(f"Predicted Class: {prediction}, Confidences: {probabilities}")
```

---

## 7. FastAPI Backend Service Architecture (Implemented — Phase 5)

The production inference service (`src/api/app.py` and `src/api/service.py`) wraps the serialized pipeline in a high-performance, asynchronous REST service:

- **Application Lifespan & Singleton Service:**
  - `WineModelService` operates as a centralized singleton loaded on application startup via FastAPI's `@asynccontextmanager` lifespan handler.
  - Model verification checks ensure the loaded artifact is an `sklearn.pipeline.Pipeline`, has steps `["StandardScaler", "SVC"]`, and feature names match the exact training schema (`13` features in order).
- **Strict Input Validation (`src/schemas/wine_schema.py`):**
  - Pydantic v2 `WineFeatures` validates all 13 chemical constituents.
  - Configured with `extra="forbid"` to strictly disallow unexpected fields.
  - Rejects booleans, `NaN`, and `Infinity` values upfront.
  - Supports the 12th feature slash notation (`od280/od315_of_diluted_wines`) via field alias.
- **REST Endpoints:**
  - `GET /health`: Liveness and readiness probe reporting status (`"healthy"` or `"unhealthy"`), `model_loaded`, `model_ready`, loaded model name, and API version. Returns `503` if model is degraded or unloaded.
  - `GET /model-info`: Returns model name, model type (`sklearn.svm.SVC`), pipeline steps (`["StandardScaler", "SVC"]`), 13 ordered feature names, 3 target class names, tuned hyperparameters, Phase 3 cross-validation metrics, and Phase 4 holdout test metrics. Sanitized to never leak host paths.
  - `POST /predict`: Single wine sample inference returning `predicted_class` (`0`, `1`, or `2`), `predicted_class_name` (`"class_0"`, `"class_1"`, `"class_2"`), class probability mapping with sum $\approx 1.0$, `model_name`, and `model_version`.
  - `POST /predict/batch`: High-throughput bulk inference accepting a list of wine samples (max 100). Preserves sample ordering and returns indexed predictions with total and successful counts.
- **Robust Error Handling:**
  - `422 Unprocessable Content`: Request validation failures return clean JSON detailing the exact parameter location and message.
  - `404 Not Found`: Standard response for unregistered routes.
  - `405 Method Not Allowed`: Handled automatically for unsupported HTTP verbs.
  - `503 Service Unavailable`: Cleanly raised if model artifact is unloaded or corrupted.
  - `500 Internal Server Error`: Sanitized generic message preventing server stack trace leakage.

---

## 8. Testing & Quality Assurance Strategy

- **Active Unit & Integration Tests (85 Tests Passing):**
  - Foundation checks (`tests/test_foundation.py` — 5 tests active).
  - Data ingestion & inspection (`tests/test_data_loader.py` — 10 tests active).
  - Preprocessing transformation & split integrity (`tests/test_preprocessing.py` — 11 tests active).
  - Model training, CV metrics, tuning & selection (`tests/test_training.py` — 13 tests active).
  - Holdout evaluation, confusion matrix & metrics (`tests/test_evaluation.py` — 9 tests active).
  - Model serialization, artifact reload & inference consistency (`tests/test_model_serialization.py` — 11 tests active).
  - FastAPI endpoints, validation, error handling & batch predictions (`tests/test_api.py` — 26 tests active).
- **Code Coverage Tooling:** `pytest-cov` reporting 76% total statement coverage (100% schemas & config, 98% app, 92% service).
- **Static Code Analysis:** `ruff check .` enforces strict formatting, import ordering, and lint standards with 0 errors across the codebase.
- **Dependency Integrity:** `python -m pip check` confirms 0 broken requirements.
- **Continuous Integration:** Automated GitHub Actions pipeline (`.github/workflows/ci.yml`) triggered on push and pull requests to `main` and `master`.

---

## 9. Containerization & Deployment Architecture (Implemented — Phase 7)

The production inference service is fully containerized using Docker and Docker Compose:

- **Base Image:** `python:3.13-slim` (official Debian-based minimal image minimizing attack surface).
- **Security & Least Privilege:**
  - Dedicated non-root user `appuser` (UID/GID `10001`).
  - Strict ownership (`/app` owned by `appuser:appuser`).
  - Zero secrets or tokens baked into image layers.
  - Exposes exclusively port `8000`.
- **Layer Caching Optimization:**
  - `requirements.txt` is copied and installed with `--no-cache-dir` in an isolated layer prior to copying source code.
  - Automated `python -m pip check` verifies dependency consistency inside the build container.
- **Build Context & Exclusions (`.dockerignore`):**
  - Excludes `.git`, `.venv`, `__pycache__`, raw/processed data sets, notebooks, documentation, and local caches.
  - Explicitly packages `src/`, `models/best_wine_pipeline.joblib`, `models/model_metadata.json`, `models/evaluation_metrics.json`, `run.py`, and `README.md`.
- **Liveness & Readiness Healthcheck:**
  - Configured with `HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3`.
  - Executes `python -c "import urllib.request, sys; res = urllib.request.urlopen('http://127.0.0.1:8000/health'); sys.exit(0 if res.getcode() == 200 else 1)"`.
  - Queries FastAPI `/health` endpoint to verify model readiness before routing traffic.
- **Docker Compose Orchestration (`docker-compose.yml`):**
  - Declares single microservice `wine-api`.
  - Port mapping: `8000:8000`.
  - Environment variables: `HOST=0.0.0.0`, `PORT=8000`.
  - Restart policy: `unless-stopped`.
- **Cloud Deployment Readiness Checklist:**
  - [x] Lean, reproducible container image definition (`Dockerfile`).
  - [x] Application health check probing actual model readiness.
  - [x] Stateless inference design (no volume locks or database dependency).
  - [x] Configuration parameterized via environment variables (`.env.example`).
  - [x] Non-root security compliance for container runtimes (AWS ECS, Google Cloud Run, Azure Container Apps, Kubernetes).

