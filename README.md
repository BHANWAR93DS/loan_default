# 💳 Loan Default Prediction System

An end-to-end **Machine Learning pipeline** for predicting loan default risk, built with production-ready components including preprocessing, experimentation, drift monitoring, and deployment using **Flask** and **Docker**.

---

## 🚀 Project Overview

This project predicts whether a client will **default on a loan** using financial and behavioral data.

It covers the complete ML lifecycle:
- Data preprocessing (EDA-driven)
- Feature engineering & selection
- Model training (Random Forest, Gradient Boosting)
- Experiment tracking (MLflow)
- Drift detection (PSI, CSI, KS)
- Model explainability (SHAP)
- Deployment (Flask + Docker)

---

## 🧠 Key Features

- ✅ EDA-aligned preprocessing pipeline
- ✅ Missing value handling (SimpleImputer - median strategy)
- ✅ Outlier clipping (1st–99th percentile IQR)
- ✅ Class imbalance handling (SMOTE)
- ✅ Feature engineering (6 new features)
- ✅ MLflow experiment tracking (2 experiments)
- ✅ Drift detection (PSI, CSI, KS)
- ✅ SHAP explainability (summary, bar, waterfall plots)
- ✅ Flask REST API for inference
- ✅ Docker containerization

---

## 📁 Project Structure

```
loan_default/
│
├── app.py              # Flask REST API
├── model.py            # Model training + MLflow logging
├── drift.py            # PSI, CSI, KS drift detection
├── explain.py          # SHAP explainability
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker container config
├── mlruns/             # MLflow experiment tracking
└── data/
    ├── Dataset.csv         # Raw dataset (121,856 rows)
    ├── model.pkl           # Trained model
    ├── scaler.pkl          # StandardScaler
    ├── feature_columns.pkl # Feature list
    ├── label_encoders.pkl  # Label encoders
    └── imputer.pkl         # SimpleImputer
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Rows | 121,856 |
| Features | 39 |
| Target | `default` (0 = No Default, 1 = Default) |
| Default Rate | 8.08% |
| Missing Values | ~395,931 |

---

## 🔧 Feature Engineering

6 new features created from existing data:

| Feature | Formula |
|---|---|
| `age_years` | `abs(age_days) / 365` |
| `employed_years` | `abs(employed_days) / 365` |
| `loan_to_income` | `credit_amount / (client_income + 1)` |
| `annuity_to_income` | `loan_annuity / (client_income + 1)` |
| `bureau_per_year` | `credit_bureau / (age_years + 1)` |
| `social_risk` | `social_circle_default / (family_members + 1)` |

---

## 📈 Model Performance

| Model | AUC | Accuracy | F1 Score |
|---|---|---|---|
| Random Forest | 0.8729 | 79.52% | 0.6284 |
| **Gradient Boosting** ✅ | **0.9210** | **91.98%** | **0.7909** |

**Best Model: Gradient Boosting (AUC = 0.921)**

---

## 🔬 MLflow Experiments

2 experiments tracked with different hyperparameters:

**Experiment 1 — Random Forest:**
- n_estimators: 100, max_depth: 10
- class_weight: balanced

**Experiment 2 — Gradient Boosting:**
- n_estimators: 200, max_depth: 5
- learning_rate: 0.05, subsample: 0.8

View MLflow UI:
```bash
mlflow ui --port 5001
# Open: http://localhost:5001
```

---

## 📉 Data Drift Detection

| Metric | Result |
|---|---|
| Overall PSI | 0.0098 ✅ No drift |
| Avg CSI | 0.0090 ✅ No drift |
| Max CSI | 0.0368 ✅ No drift |
| Drifted Features | 0 / 44 |

PSI Thresholds:
- `< 0.10` → No significant drift ✅
- `0.10–0.25` → Moderate drift ⚠️
- `> 0.25` → Significant drift 🚨

---

## 🔍 SHAP Explainability

Top 10 features driving default predictions:

| Rank | Feature | SHAP Importance |
|---|---|---|
| 1 | application_process_day | 0.4079 |
| 2 | score_source_3 | 0.3145 |
| 3 | score_source_2 | 0.3029 |
| 4 | client_education | 0.1813 |
| 5 | client_gender | 0.1617 |
| 6 | credit_bureau | 0.1372 |
| 7 | car_owned | 0.1196 |
| 8 | client_occupation | 0.0991 |
| 9 | score_source_1 | 0.0953 |
| 10 | credit_amount | 0.0705 |

---

## ⚙️ Installation

```bash
git clone https://github.com/BHANWAR93DS/loan_default.git
cd loan_default
pip install -r requirements.txt
```

---

## ▶️ Run Pipeline

```bash
# Step 1: Train model + log to MLflow
python model.py

# Step 2: SHAP explainability
python explain.py

# Step 3: Drift detection
python drift.py

# Step 4: View MLflow UI
mlflow ui --port 5001

# Step 5: Run Flask API
python app.py
```

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t loan-default .

# Run
docker run -p 5000:5000 loan-default

# Test
curl http://localhost:5000/health
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/features` | Feature list |
| GET | `/sample` | Sample request |
| POST | `/predict` | Predict default |

### Sample Request:
```bash
POST http://localhost:5000/predict
Content-Type: application/json

{
  "client_income": 180000,
  "credit_amount": 450000,
  "loan_annuity": 20250,
  "age_days": -12005,
  "employed_days": -2500,
  "score_source_2": 0.6,
  "credit_bureau": 1
}
```

### Sample Response:
```json
{
  "default_probability": 0.1823,
  "prediction": 0,
  "risk_level": "Low",
  "message": "No default expected — low risk client"
}
```

---

## 🛠️ Tech Stack

- Python 3.10
- Pandas, NumPy, Scikit-learn
- Imbalanced-learn (SMOTE)
- MLflow
- SHAP
- Flask
- Docker
- Scipy

---

## 👤 Author

**Bhanwar Lal**

---

⭐ If you found this useful, give it a star!
