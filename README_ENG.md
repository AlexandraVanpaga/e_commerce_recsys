# E-Commerce Recommendation System

A recommendation system for an e-commerce platform. The goal is to predict which products to suggest to online store users.

## Overview

The system builds a full recommendation pipeline: from data analysis and feature engineering to model training, deployment as a web service, and automated retraining via Airflow.

### Key Features:
- Candidate generation via ALS (200 items)
- Reranking with CatBoost using behavioral and product features
- Fallback strategy for cold-start users (top-100 popular products)
- Dockerized microservice with health checks
- Monitoring via Prometheus + Grafana
- Automated retraining pipeline with Airflow + DVC versioning

---

## Project Structure

```
e_commerce_recsys/
├── data/
│   ├── archive/                  # Raw data
│   └── processed/                # Processed data
├── models/                       # Saved models
├── src/
│   ├── get_raw_data.py           # Raw data download
│   ├── process_data.py           # Data preprocessing
│   ├── als_and_ranking.py        # ALS + CatBoost training
│   ├── prepare_features.py       # Feature preparation (similar_items, top_products, item_features)
│   ├── eval.py                   # Model evaluation
│   └── test_script.py            # Load testing
├── airflow/
│   └── dags/
│       └── recsys_retrain.py     # Retraining DAG
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Installation

### Requirements

- Python 3.8+
- Docker & Docker Compose
- CUDA (optional, for GPU)

### Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Download Raw Data

```bash
python -m src.get_raw_data
```

### 2. Data Preprocessing

```bash
python -m src.process_data
```

Performs:
- Duplicate event removal
- Column renaming for clarity
- Extracting category info from the parent category tree
- Versioning cleaned data via DVC

### 3. Feature Preparation

```bash
python -m src.prepare_features
```

Generates features: `similar_items`, `top_products`, `item_features`

### 4. Model Training

```bash
# Train ALS + CatBoost Ranker
python -m src.als_and_ranking
```

### 5. Model Evaluation

```bash
python -m src.eval
```

### 6. Launch Microservice

```bash
cd ~/mle_projects/e_commerce_recsys
docker compose up --build
```

### 7. Health Checks

```bash
# API
curl http://localhost:8081/health

# Prometheus
curl http://localhost:9090/-/healthy

# Grafana
curl http://localhost:3000/api/health
```

---

## API

### Get Recommendations

```bash
curl -X POST http://localhost:8081/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123, "k": 20}'
```

### Load Testing

```bash
python -m src.test_script
```

---

## How the Recommendation Strategy Works

1. Service receives `user_id`
2. Generates candidates via ALS (200 items)
3. Reranks them with CatBoost using the following features:
   - Product popularity
   - Categories and parent categories
   - User activity
   - Action type weights (`view=1`, `cart=5`, `purchase=10`)
   - Category match with user preferences
4. Returns top-K recommendations

**Fallback strategy:** If no ALS candidates are available for a user (cold-start), the system returns top-100 popular products.

---

## 📈 Results

### ALS

| Top-N | Recall | Precision |
|-------|--------|-----------|
| @10 | 0.0065 | 0.0014 |
| @20 | 0.0087 | 0.0010 |
| @50 | 0.0131 | 0.0006 |
| @100 | 0.0175 | 0.0004 |

Shows steady Recall growth and expected Precision drop as recommendation list size increases.

### CatBoost Ranker

| Top-N | Recall | Precision |
|-------|--------|-----------|
| @10 | 0.0055 | 0.0011 |
| @20 | 0.0073 | 0.0008 |
| @50 | 0.0122 | 0.0006 |
| @100 | 0.0179 | 0.0004 |

CatBoost Ranker shows comparable quality to ALS.

### Conclusions

Both models are suitable as recommendation baselines. CatBoost offers an advantage in extensibility (incorporating additional features), while ALS serves as a simple and fast baseline. Retraining on the full dataset yields slightly better results due to more data, but improvements are marginal and consistent with validation results.

---

## 🔧 Monitoring

**Grafana data source:** `http://prometheus:9090`

**Dashboards:**
- JSON: `rec_sys_dashboard.json`
- Preview: `rec_sys_dashboard.jpg`

---

## Retraining Pipeline

### Strategy

To simulate a real-world scenario, 10% of data is reserved (events after `2015-09-01`). In production, this data is merged with historical data and the model is retrained via Airflow.

### Launch Airflow

```bash
cd airflow

# Initialize
docker compose up airflow-init

# Clean up (if needed)
docker compose down --volumes --remove-orphans

# Start
docker compose up --build
```

**DAG name:** `recsys_retrain`

**Trigger:** Manually via Airflow UI (`http://localhost:8080`)

**Pipeline steps:**
1. `retrain_als` — retrain ALS on new data
2. `prepare_features` — recalculate features
3. `generate_recommendations` — generate ALS candidates
4. `train_ranker` — train CatBoost ranker with quality evaluation

---

## Technologies

- **ML:** implicit (ALS), CatBoost (ranking), scikit-learn
- **Data:** pandas, numpy, scipy
- **MLOps:** MLflow, DVC, Airflow
- **API:** FastAPI, uvicorn
- **Monitoring:** Prometheus, Grafana
- **Infrastructure:** Docker, Docker Compose, PostgreSQL, S3
EOF
Salida

# E-Commerce Recommendation System

A recommendation system for an e-commerce platform. The goal is to predict which products to suggest to online store users.

## Overview

The system builds a full recommendation pipeline: from data analysis and feature engineering to model training, deployment as a web service, and automated retraining via Airflow.

### Key Features:
- Candidate generation via ALS (200 items)
- Reranking with CatBoost using behavioral and product features
- Fallback strategy for cold-start users (top-100 popular products)
- Dockerized microservice with health checks
- Monitoring via Prometheus + Grafana
- Automated retraining pipeline with Airflow + DVC versioning

---

## Project Structure

```
e_commerce_recsys/
├── data/
│   ├── archive/                  # Raw data
│   └── processed/                # Processed data
├── models/                       # Saved models
├── src/
│   ├── get_raw_data.py           # Raw data download
│   ├── process_data.py           # Data preprocessing
│   ├── als_and_ranking.py        # ALS + CatBoost training
│   ├── prepare_features.py       # Feature preparation (similar_items, top_products, item_features)
│   ├── eval.py                   # Model evaluation
│   └── test_script.py            # Load testing
├── airflow/
│   └── dags/
│       └── recsys_retrain.py     # Retraining DAG
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Installation

### Requirements

- Python 3.8+
- Docker & Docker Compose
- CUDA (optional, for GPU)

### Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Download Raw Data

```bash
python -m src.get_raw_data
```

### 2. Data Preprocessing

```bash
python -m src.process_data
```

Performs:
- Duplicate event removal
- Column renaming for clarity
- Extracting category info from the parent category tree
- Versioning cleaned data via DVC

### 3. Feature Preparation

```bash
python -m src.prepare_features
```

Generates features: `similar_items`, `top_products`, `item_features`

### 4. Model Training

```bash
# Train ALS + CatBoost Ranker
python -m src.als_and_ranking
```

### 5. Model Evaluation

```bash
python -m src.eval
```

### 6. Launch Microservice

```bash
cd ~/mle_projects/e_commerce_recsys
docker compose up --build
```

### 7. Health Checks

```bash
# API
curl http://localhost:8081/health

# Prometheus
curl http://localhost:9090/-/healthy

# Grafana
curl http://localhost:3000/api/health
```

---

## API

### Get Recommendations

```bash
curl -X POST http://localhost:8081/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123, "k": 20}'
```

### Load Testing

```bash
python -m src.test_script
```

---

## How the Recommendation Strategy Works

1. Service receives `user_id`
2. Generates candidates via ALS (200 items)
3. Reranks them with CatBoost using the following features:
   - Product popularity
   - Categories and parent categories
   - User activity
   - Action type weights (`view=1`, `cart=5`, `purchase=10`)
   - Category match with user preferences
4. Returns top-K recommendations

**Fallback strategy:** If no ALS candidates are available for a user (cold-start), the system returns top-100 popular products.

---

## 📈 Results

### ALS

| Top-N | Recall | Precision |
|-------|--------|-----------|
| @10 | 0.0065 | 0.0014 |
| @20 | 0.0087 | 0.0010 |
| @50 | 0.0131 | 0.0006 |
| @100 | 0.0175 | 0.0004 |

Shows steady Recall growth and expected Precision drop as recommendation list size increases.

### CatBoost Ranker

| Top-N | Recall | Precision |
|-------|--------|-----------|
| @10 | 0.0055 | 0.0011 |
| @20 | 0.0073 | 0.0008 |
| @50 | 0.0122 | 0.0006 |
| @100 | 0.0179 | 0.0004 |

CatBoost Ranker shows comparable quality to ALS.

### Conclusions

Both models are suitable as recommendation baselines. CatBoost offers an advantage in extensibility (incorporating additional features), while ALS serves as a simple and fast baseline. Retraining on the full dataset yields slightly better results due to more data, but improvements are marginal and consistent with validation results.

---

## 🔧 Monitoring

**Grafana data source:** `http://prometheus:9090`

**Dashboards:**
- JSON: `rec_sys_dashboard.json`
- Preview: `rec_sys_dashboard.jpg`

---

## Retraining Pipeline

### Strategy

To simulate a real-world scenario, 10% of data is reserved (events after `2015-09-01`). In production, this data is merged with historical data and the model is retrained via Airflow.

### Launch Airflow

```bash
cd airflow

# Initialize
docker compose up airflow-init

# Clean up (if needed)
docker compose down --volumes --remove-orphans

# Start
docker compose up --build
```

**DAG name:** `recsys_retrain`

**Trigger:** Manually via Airflow UI (`http://localhost:8080`)

**Pipeline steps:**
1. `retrain_als` — retrain ALS on new data
2. `prepare_features` — recalculate features
3. `generate_recommendations` — generate ALS candidates
4. `train_ranker` — train CatBoost ranker with quality evaluation

---

## Technologies

- **ML:** implicit (ALS), CatBoost (ranking), scikit-learn
- **Data:** pandas, numpy, scipy
- **MLOps:** MLflow, DVC, Airflow
- **API:** FastAPI, uvicorn
- **Monitoring:** Prometheus, Grafana
- **Infrastructure:** Docker, Docker Compose, PostgreSQL, S3
