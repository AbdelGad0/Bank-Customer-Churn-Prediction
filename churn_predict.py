import json
import os

import joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_xgb_smote.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "onehot_encoder.joblib")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")

_pipeline = joblib.load(MODEL_PATH)
_encoder = joblib.load(ENCODER_PATH)
with open(CONFIG_PATH) as f:
    _config = json.load(f)

THRESHOLD = _config["threshold"]
NUMERIC_ORDER = _config["numeric_order"]
CAT_COLS = _config["cat_cols"]
ENCODED_COLS = [c for c in _config["feature_columns"] if c not in NUMERIC_ORDER]


def preprocess(geography, gender, age, balance, num_of_products, is_active_member):
    raw = pd.DataFrame(
        {
            "Geography": [geography],
            "Gender": [gender],
            "Age": [age],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "IsActiveMember": [is_active_member],
        }
    )
    return raw


def predict_batch(raw_df):
    numeric = raw_df[NUMERIC_ORDER].astype(float).reset_index(drop=True)
    enc = pd.DataFrame(
        _encoder.transform(raw_df[CAT_COLS]),
        columns=ENCODED_COLS,
    )
    X = pd.concat([numeric, enc], axis=1)
    return _pipeline.predict_proba(X)[:, 1]


def predict_churn(geography, gender, age, balance=0.0, num_of_products=1, is_active_member=1):
    raw = preprocess(geography, gender, age, balance, num_of_products, is_active_member)
    proba = float(predict_batch(raw)[0])
    return {
        "probability": round(proba, 4),
        "prediction": int(proba >= THRESHOLD),
        "threshold": THRESHOLD,
        "risk": "high" if proba >= THRESHOLD else "low",
    }


if __name__ == "__main__":
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        roc_auc_score,
        average_precision_score,
    )

    demo = predict_churn(
        geography="Germany", gender="Male", age=45,
        balance=100000, num_of_products=2, is_active_member=0,
    )
    print("Demo prediction:", demo)

    # verify on full test set (reconstruct raw test rows -> same split seed)
    from sklearn.model_selection import train_test_split

    raw = pd.read_csv(os.path.join(BASE, "Churn_Modelling_cleaned.csv"))
    X_raw = raw.drop(columns=["Exited"])
    y_raw = raw["Exited"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_raw, y_raw, test_size=0.30, stratify=y_raw, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    rows = []
    for _, row in X_test.iterrows():
        r = predict_churn(
            geography=row["Geography"], gender=row["Gender"], age=row["Age"],
            balance=row["Balance"], num_of_products=row["NumOfProducts"],
            is_active_member=row["IsActiveMember"],
        )
        rows.append(r["probability"])
    proba = np.array(rows)
    y_pred = (proba >= THRESHOLD).astype(int)
    print("\nVerification on test set (saved model):")
    print(f"  Accuracy = {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision= {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall   = {recall_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  = {roc_auc_score(y_test, proba):.4f}")
    print(f"  PR-AUC   = {average_precision_score(y_test, proba):.4f}")
    print(f"\nExpected (champion)     : Acc=0.7493 Prec=0.4385 Rec=0.8295 ROC=0.8695 PR=0.7299")