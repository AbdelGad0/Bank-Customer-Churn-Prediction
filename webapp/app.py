import io
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from flask import Flask, jsonify, make_response, render_template, request

from churn_predict import predict_churn, predict_batch, THRESHOLD

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

REQUIRED = ["Geography", "Gender", "Age"]
NUMBER_FIELDS = {"Age": "float", "Balance": "float", "NumOfProducts": "int", "IsActiveMember": "int"}

COLUMN_ALIASES = {
    "geography": "Geography", "country": "Geography",
    "gender": "Gender", "sex": "Gender",
    "age": "Age", "ageyears": "Age",
    "balance": "Balance",
    "numofproducts": "NumOfProducts", "num_products": "NumOfProducts", "products": "NumOfProducts",
    "isactivemember": "IsActiveMember", "activemember": "IsActiveMember", "active": "IsActiveMember",
}
GEO_MAP = {"france": "France", "germany": "Germany", "spain": "Spain",
           "fr": "France", "de": "Germany", "es": "Spain"}
GENDER_MAP = {"male": "Male", "female": "Female", "m": "Male", "f": "Female",
              "ذكر": "Male", "أنثى": "Female"}


def clean_col(c):
    return re.sub(r"[\s\-_]+", "", str(c).strip().lower())


def normalize_table(df):
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    renamed = {}
    for col in df.columns:
        key = clean_col(col)
        if key in COLUMN_ALIASES:
            renamed[col] = COLUMN_ALIASES[key]
    df = df.rename(columns=renamed)
    present = list(dict.fromkeys(c for c in REQUIRED + list(NUMBER_FIELDS) if c in df.columns))
    extras = [c for c in df.columns if c not in present]
    df = df[present + extras]
    return df


def coerce_values(df):
    errs = []
    for col, ctype in NUMBER_FIELDS.items():
        if col not in df.columns:
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if ctype == "int":
                df[col] = df[col].fillna(0).astype(int)
            else:
                df[col] = df[col].fillna(0.0)
        except Exception:
            errs.append(col)
    if "Geography" in df.columns:
        df["Geography"] = df["Geography"].astype(str).str.strip().str.lower().map(GEO_MAP).fillna("France")
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip().str.lower().map(GENDER_MAP).fillna("Male")
    for col in ["Balance", "NumOfProducts", "IsActiveMember"]:
        if col not in df.columns:
            df[col] = 1 if col == "NumOfProducts" else (0.0 if col == "Balance" else 1)
    return df, errs


def read_table(path, filename):
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext in ("xlsx", "xls"):
        return pd.read_excel(path)
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    return pd.read_csv(path, engine="python")


def rows_to_json(df):
    return [{c: (None if pd.isna(v) else v) for c, v in r.items()}
            for r in df.to_dict("records")]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "threshold": THRESHOLD})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    try:
        res = predict_churn(
            geography=data["geography"],
            gender=data["gender"],
            age=float(data["age"]),
            balance=float(data.get("balance", 0.0)),
            num_of_products=int(data.get("num_of_products", 1)),
            is_active_member=int(data.get("is_active_member", 1)),
        )
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e.args[0]}"}), 400
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400
    return jsonify(res)


@app.route("/api/predict_file", methods=["POST"])
def api_predict_file():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file uploaded"}), 400
    ext = file.filename.lower().rsplit(".", 1)[-1]
    if ext not in ("csv", "xlsx", "xls"):
        return jsonify({"error": "Supported formats: .csv, .xlsx, .xls"}), 400
    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)
    try:
        df = read_table(path, file.filename)
        df = normalize_table(df)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            return jsonify({"error": "Missing columns: " + ", ".join(missing)}), 400
        df, errs = coerce_values(df)
        df["Geography"] = df["Geography"].astype(str)
        df["Gender"] = df["Gender"].astype(str)
        proba = predict_batch(df)
        df["Churn_Probability"] = proba.round(4)
        df["Churn_Prediction"] = (proba >= THRESHOLD).astype(int)
        df["Risk"] = df["Churn_Prediction"].map({1: "high", 0: "low"})
        high = int(df["Churn_Prediction"].sum())
        total = int(len(df))
        return jsonify({
            "total": total,
            "high_risk": high,
            "high_risk_pct": round(high / total * 100, 2) if total else 0,
            "threshold": THRESHOLD,
            "columns": [str(c) for c in df.columns],
            "rows": rows_to_json(df),
            "errors": errs,
        })
    except Exception as e:
        return jsonify({"error": f"Processing failed: {e}"}), 400
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.route("/api/template")
def template():
    sample = pd.DataFrame(
        [
            {"Geography": "France", "Gender": "Female", "Age": 42, "Balance": 0,
             "NumOfProducts": 1, "IsActiveMember": 1, "EstimatedSalary": 101348.88},
            {"Geography": "Germany", "Gender": "Male", "Age": 55, "Balance": 120000,
             "NumOfProducts": 2, "IsActiveMember": 0, "EstimatedSalary": 85000.0},
            {"Geography": "Spain", "Gender": "Female", "Age": 33, "Balance": 45000,
             "NumOfProducts": 2, "IsActiveMember": 1, "EstimatedSalary": 92000.0},
        ]
    )
    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    resp = make_response("\ufeff" + buf.getvalue())
    resp.headers["Content-Disposition"] = "attachment; filename=template.csv"
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)