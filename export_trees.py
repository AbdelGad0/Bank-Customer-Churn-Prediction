import json
import os
import io

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

    # base_score
    bcfg = json.loads(booster.save_config())
    raw_bs = bcfg["learner"]["learner_model_param"]["base_score"]
    if isinstance(raw_bs, str):
        raw_bs = raw_bs.strip("[]")
    base_score = float(raw_bs)
    feature_names = booster.feature_names

    # tree dump (list of root nodes, one per tree)
    buf = io.StringIO()
    booster.dump_model(fout=buf, dump_format="json")
    roots = json.loads(buf.getvalue())

    trees = []
    for root in roots:
        max_nid = 0
        queue = [root]
        while queue:
            n = queue.pop(0)
            max_nid = max(max_nid, n["nodeid"])
            queue.extend(n.get("children", []))
        size = max_nid + 1

        fi = [0] * size
        th = [0.0] * size
        yes = [0] * size
        no = [0] * size
        missing = [0] * size
        leaf_value = [0.0] * size
        is_leaf = [False] * size

        queue = [root]
        while queue:
            n = queue.pop(0)
            nid = n["nodeid"]
            if "leaf" in n:
                is_leaf[nid] = True
                leaf_value[nid] = float(n["leaf"])
            else:
                fi[nid] = feature_names.index(n["split"])
                th[nid] = float(n["split_condition"])
                yes[nid] = n["yes"]
                no[nid] = n["no"]
                missing[nid] = n.get("missing", n["yes"])
                queue.extend(n.get("children", []))

        trees.append({
            "feature_idx": fi,
            "threshold": th,
            "yes": yes,
            "no": no,
            "missing": missing,
            "leaf_value": leaf_value,
            "is_leaf": is_leaf,
        })

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

    out = {"base_score": base_score, "feature_names": feature_names, "num_trees": len(trees), "trees": trees}
    with open(os.path.join(MODEL_DIR, "churn_xgb_trees.json"), "w") as f:
        json.dump(out, f)

    print(f"Exported {len(trees)} trees, base_score={base_score}, features={len(feature_names)}")


if __name__ == "__main__":
    main()