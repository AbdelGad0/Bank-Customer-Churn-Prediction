import json
import os

import joblib

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")


def main():
    pipe = joblib.load(os.path.join(MODEL_DIR, "churn_xgb_smote.joblib"))
    enc = joblib.load(os.path.join(MODEL_DIR, "onehot_encoder.joblib"))
    cfg_path = os.path.join(MODEL_DIR, "model_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    booster = pipe.steps[-1][1].get_booster()
    booster.save_model(os.path.join(MODEL_DIR, "churn_xgb_slim.model"))

    cat_cols = cfg["cat_cols"]
    cats_per_col = {col: list(cats) for col, cats in zip(cat_cols, enc.categories_)}
    numeric = set(cfg["numeric_order"])
    onehot_map = []
    for feat in cfg["feature_columns"]:
        if feat in numeric:
            continue
        for col in cat_cols:
            for cat in cats_per_col[col]:
                if f"{col}_{cat}" == feat:
                    onehot_map.append({"feature": feat, "col": col, "cat": cat})
    cfg["onehot_map"] = onehot_map
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("slim saved:", os.path.join(MODEL_DIR, "churn_xgb_slim.model"))
    print("onehot_map:", onehot_map)


if __name__ == "__main__":
    main()