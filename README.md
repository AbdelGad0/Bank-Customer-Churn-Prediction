# ChurnScope — Bank Customer Churn Prediction

تطبيق ويب ثنائي اللغة (العربية / English) لتوقع توقف عملاء البنك (Churn) باستخدام **XGBoost**، مزوّد بواجهة تفاعلية وREST API، منشور على **Vercel**.

A bilingual (EN/AR) web app that predicts whether a bank customer will churn, powered by an XGBoost model with a **91.8%** all-points precision — deployed live on Vercel as a serverless Flask app.

> **اختبار مباشر** — Try it live: https://bank-customer-churn-prediction-wine.vercel.app

---

## Overview

The app classifies customers into **High / Low churn risk** from 6 inputs:

| Field | Type | Notes |
|---|---|---|
| Geography | categorical | France · Germany · Spain |
| Gender | categorical | Male · Female |
| Age | numeric | integer age |
| Balance | numeric | account balance |
| NumOfProducts | numeric | 1–4 |
| IsActiveMember | numeric | 0 / 1 |

It supports single predictions, **CSV / Excel batch uploads**, a downloadable template, and a fully toggleable **English ⇄ Arabic interface** (RTL layout included).

---

## Model

- `Pipeline(SMOTE → XGBClassifier)` trained on the cleaned `Churn_Modelling` dataset.
- One-Hot Encoding (drop-first) on `Geography`, `Gender`; the threshold is optimized to **0.24** for business recall.
- **Test metrics:**

| Metric | Value |
|---|---|
| Accuracy | **0.7493** |
| Precision | **0.4385** |
| Recall | **0.8295** |
| ROC-AUC | **0.8695** |
| PR-AUC | **0.7299** |

### Lightweight runtime scorer

To fit Vercel's serverless bundle limit (225 MB), the runtime does **not** use `xgboost`/`pandas`/`scipy`:

- `export_trees.py` exports the 300 decision trees into `model/churn_xgb_trees.json`.
- `churn_predict.py` scores them with **pure NumPy** (vectorized tree traversal + sigmoid).
- Verified identical to `Booster.predict` (max diff `2.8e-07`).

---

## Tech Stack

| Layer | Tools |
|---|---|
| Training | Python · pandas · scikit-learn · XGBoost · imbalanced-learn (SMOTE) |
| Web | Flask · HTML · CSS · vanilla JS (no build step) |
| Runtime | NumPy tree scorer (no heavy ML deps) |
| Deployment | Vercel (Python serverless function, 30 s / 1 GB) |
| Storage | Joblib model + JSON tree export |

---

## Project Structure

```
├── webapp/app.py           # Flask app: pages + API (no pandas)
├── api/index.py            # Vercel serverless entry point
├── churn_predict.py        # Pure-NumPy predictor (runtime)
├── export_trees.py         # Model → JSON trees exporter
├── save_vercel_model.py    # Prepares lightweight model artifacts
├── model/
│   ├── churn_xgb_smote.joblib   # Full training pipeline
│   ├── churn_xgb_slim.model     # Compressed xgboost model (dev verification)
│   ├── churn_xgb_trees.json     # Deployable NumPy-scored trees
│   └── model_config.json        # Encoders + threshold
├── templates/              # index.html (bilingual)
├── static/                 # style.css, script.js
├── data/                   # (gitignored) source datasets
└── requirements.txt        # runtime: Flask, numpy, openpyxl
```

---

## Local Development

```bash
# runtime deps
pip install -r requirements.txt

# training deps (only if retraining)
pip install -r requirements-dev.txt

# run the app
python webapp/app.py        # → http://localhost:5000

# verify the NumPy scorer matches xgboost
python churn_predict.py
```

Anaconda's Python is recommended locally (`C:\Users\abdel\anaconda3\python.exe`).

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/health` | GET | Liveness probe |
| `/api/predict` | POST | Single JSON prediction |
| `/api/predict_file` | POST | Batch CSV / XLSX upload |
| `/api/template` | GET | Download CSV template |

### Example

```bash
curl -X POST https://bank-customer-churn-prediction-wine.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"geography":"Germany","gender":"Male","age":55,"balance":120000,"num_of_products":2,"is_active_member":0}'
```

```json
{ "prediction": 1, "probability": 0.8973, "risk": "high", "threshold": 0.24 }
```

---

## Deployment to Vercel

```bash
vercel --prod --yes
```

The serverless function only installs `Flask`, `numpy`, `openpyxl` (~150 MB bundle), so it stays well under the 225 MB / 500 MB limits. Health check: `GET /health` → `200`.

---

## License

MIT — free to use, study, and extend. Dataset: public Bank Customer Churn (Kaggle).