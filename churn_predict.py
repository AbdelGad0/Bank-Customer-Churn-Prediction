import json
import os

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")
TREES_PATH = os.path.join(MODEL_DIR, "churn_xgb_trees.json")

with open(CONFIG_PATH) as f:
    _config = json.load(f)

THRESHOLD = _config["threshold"]
NUMERIC_ORDER = _config["numeric_order"]
FEATURE_COLUMNS = _config["feature_columns"]
ENCODED_COLS = [c for c in FEATURE_COLUMNS if c not in NUMERIC_ORDER]
ONHOT_MAP = [(m["feature"], m["col"], m["cat"]) for m in _config["onehot_map"]]
NUMERIC_DEFAULTS = {"Age": 0.0, "Balance": 0.0, "NumOfProducts": 1, "IsActiveMember": 1}

with open(TREES_PATH) as f:
    _model = json.load(f)

_BASE_SCORE = _model["base_score"]
_INIT_MARGIN = float(np.log(_BASE_SCORE / (1.0 - _BASE_SCORE))) if 0.0 < _BASE_SCORE < 1.0 else 0.0
_TREES = []
for tree in _model["trees"]:
    _TREES.append({
        "feature_idx": np.asarray(tree["feature_idx"], dtype=np.int32),
        "threshold": np.asarray(tree["threshold"], dtype=np.float64),
        "yes": np.asarray(tree["yes"], dtype=np.int32),
        "no": np.asarray(tree["no"], dtype=np.int32),
        "leaf_value": np.asarray(tree["leaf_value"], dtype=np.float64),
        "is_leaf": np.asarray(tree["is_leaf"], dtype=bool),
    })


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def predict_batch(raw_rows):
    n = len(raw_rows)
    numeric = np.array(
        [[float(row.get(k, NUMERIC_DEFAULTS[k])) for k in NUMERIC_ORDER] for row in raw_rows],
        dtype=np.float64,
    )
    enc = np.array(
        [
            [1.0 if str(row.get(col, "")).strip().lower() == cat.lower() else 0.0
             for feat, col, cat in ONHOT_MAP]
            for row in raw_rows
        ],
        dtype=np.float64,
    )
    X = np.concatenate([numeric, enc], axis=1)

    score = np.full(n, _INIT_MARGIN, dtype=np.float64)
    for tree in _TREES:
        fi = tree["feature_idx"]
        th = tree["threshold"]
        yes = tree["yes"]
        no_arr = tree["no"]
        lf = tree["leaf_value"]
        is_leaf = tree["is_leaf"]

        node = np.zeros(n, dtype=np.int32)
        for _ in range(10):
            if is_leaf[node].all():
                break
            mask = ~is_leaf[node]
            vals = X[mask, fi[node[mask]]]
            left = vals < th[node[mask]]
            node[mask] = np.where(left, yes[node[mask]], no_arr[node[mask]])

        score += lf[node]

    return _sigmoid(score)


def predict_churn(geography, gender, age, balance=0.0, num_of_products=1, is_active_member=1):
    row = {
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "IsActiveMember": is_active_member,
    }
    proba = float(predict_batch([row])[0])
    return {
        "probability": round(proba, 4),
        "prediction": int(proba >= THRESHOLD),
        "threshold": THRESHOLD,
        "risk": "high" if proba >= THRESHOLD else "low",
    }


if __name__ == "__main__":
    import pandas as pd  # noqa: F401

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from xgboost import DMatrix, Booster

    bst = Booster()
    bst.load_model(os.path.join(MODEL_DIR, "churn_xgb_slim.model"))

    demo = predict_churn(
        geography="Germany", gender="Male", age=45,
        balance=100000, num_of_products=2, is_active_member=0,
    )
    print("Demo prediction:", demo)

    # verify against xgboost on equivalent inputs
    rng = np.random.RandomState(0)
    geos = ["Germany", "France", "Spain"]
    genders = ["Male", "Female"]
    test_rows = [
        {
            "Geography": rng.choice(geos),
            "Gender": rng.choice(genders),
            "Age": float(18 + rng.rand() * 70),
            "Balance": float(rng.rand() * 150000),
            "NumOfProducts": int(rng.randint(1, 5)),
            "IsActiveMember": int(rng.randint(0, 2)),
        }
        for _ in range(2000)
    ]
    my_pred = predict_batch(test_rows)
    rows_mat = np.array(
        [
            [float(r["Age"]), float(r["Balance"]), float(r["NumOfProducts"]), float(r["IsActiveMember"]),
             1.0 if r["Geography"].lower() == "germany" else 0.0,
             1.0 if r["Geography"].lower() == "spain" else 0.0,
             1.0 if r["Gender"].lower() == "male" else 0.0]
            for r in test_rows
        ],
        dtype=np.float64,
    )
    dmat = DMatrix(rows_mat, feature_names=FEATURE_COLUMNS)
    xb_pred = bst.predict(dmat)
    max_diff = float(np.max(np.abs(xb_pred - my_pred)))
    print(f"Booster vs tree max diff: {max_diff:.2e}")

    # verify on full test set
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
    print("\nVerification on test set (manual tree scorer):")
    print(f"  Accuracy = {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision= {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall   = {recall_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  = {roc_auc_score(y_test, proba):.4f}")
    print(f"  PR-AUC   = {average_precision_score(y_test, proba):.4f}")
    print(f"\nExpected (champion)     : Acc=0.7493 Prec=0.4385 Rec=0.8295 ROC=0.8695 PR=0.7299")