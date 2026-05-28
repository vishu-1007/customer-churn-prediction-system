# Customer Churn Prediction System

**Author:** Goli Raghu Sharan Teja  
**Stack:** Python · XGBoost · SHAP · Streamlit · Optuna · Scikit-learn  
**Dataset:** IBM Telco Customer Churn (Kaggle, 7k records)

---

## Folder Structure

```
churn_prediction/
├── data/
│   ├── raw/                    ← Place downloaded CSV here
│   └── processed/              ← Auto-generated after preprocessing
├── src/
│   ├── logger.py               ← Centralized logging
│   ├── utils.py                ← Config loader, model versioning
│   ├── data_loader.py          ← Data ingestion + schema validation
│   ├── preprocess.py           ← Cleaning, encoding, SMOTE, scaling
│   ├── train.py                ← LR + RF + XGBoost + Optuna tuning
│   ├── evaluate.py             ← Metrics, plots, SHAP analysis
│   └── predict.py              ← Single + batch inference
├── app/
│   └── streamlit_app.py        ← Interactive dashboard
├── models/                     ← Saved models (versioned)
├── outputs/
│   ├── plots/                  ← All charts
│   └── reports/                ← CSV/JSON comparison reports
├── configs/
│   └── config.yaml             ← All project settings
├── logs/                       ← Daily rotating log files
├── notebooks/                  ← Exploratory notebooks (optional)
├── run_pipeline.py             ← Single-command training pipeline
├── requirements.txt
└── Dockerfile
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
Save as: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### 3. Run full pipeline (preprocessing + training + evaluation)
```bash
python run_pipeline.py
```

### 4. Launch Streamlit app
```bash
streamlit run app/streamlit_app.py
```

Open: http://localhost:8501

---

## Run individual modules

```bash
# Data loading only
python src/data_loader.py

# Preprocessing only
python src/preprocess.py

# Training only (after preprocessing)
python src/train.py

# Evaluation only (after training)
python src/evaluate.py

# Single prediction test
python src/predict.py
```

---

## Deployment

### A. Streamlit Cloud
1. Push this repo to GitHub
2. Go to https://share.streamlit.io
3. Select repo → set main file: `app/streamlit_app.py`
4. Deploy

### B. Docker
```bash
# Build image
docker build -t churn-prediction .

# Run container
docker run -p 8501:8501 churn-prediction

# Open: http://localhost:8501
```

### C. Render / Railway
1. Push to GitHub
2. Connect repo on Render/Railway
3. Set start command: `streamlit run app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
4. Deploy

---

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | data_loader.py | Load CSV, validate schema, fix TotalCharges |
| 2 | preprocess.py | Clean, encode, engineer features |
| 3 | preprocess.py | Train/test split (stratified, 80/20) |
| 4 | preprocess.py | StandardScaler + SMOTE (train only) |
| 5 | train.py | Logistic Regression baseline |
| 6 | train.py | Random Forest |
| 7 | train.py | XGBoost default |
| 8 | train.py | XGBoost tuned — Optuna (50 trials) |
| 9 | evaluate.py | F1, AUC, Precision, Recall, Confusion Matrix |
| 10 | evaluate.py | SHAP summary + waterfall plots |

---

## Model Versioning

Every trained model is saved with a timestamp:
```
models/xgboost_tuned_20240601_1423.pkl
models/xgboost_tuned_20240602_0910.pkl   ← newer run
models/best_model.pkl                    ← always the latest best
```

The app always loads `models/best_model.pkl`.

---

## Results (expected)

| Model | F1 | ROC-AUC |
|-------|----|---------|
| Logistic Regression | ~76% | ~84% |
| Random Forest | ~82% | ~88% |
| XGBoost (default) | ~84% | ~89% |
| XGBoost (Optuna tuned) | ~87% | ~91% |

---

## Key Design Decisions

- **SMOTE on train only** — prevents data leakage into test evaluation
- **Optuna over GridSearchCV** — Bayesian search is 3–5x faster for same quality
- **SHAP TreeExplainer** — exact (not approximate) SHAP values for tree models
- **Modular src/** — each file has one job; easy to test or swap components
- **config.yaml** — all hyperparameters in one place, no hardcoding in code
- **Versioned models** — old models never overwritten; always reproducible
