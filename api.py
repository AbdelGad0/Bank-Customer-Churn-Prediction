from flask import Flask, jsonify, request

from churn_predict import predict_churn, THRESHOLD

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "threshold": THRESHOLD})


@app.route("/predict", methods=["POST"])
def predict():
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


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(force=True)
    rows = data.get("customers", [])
    if not isinstance(rows, list) or not rows:
        return jsonify({"error": "Expected {'customers': [...]}"}), 400
    results = []
    for row in rows:
        try:
            results.append({"input": row, **predict_churn(
                geography=row["geography"],
                gender=row["gender"],
                age=float(row["age"]),
                balance=float(row.get("balance", 0.0)),
                num_of_products=int(row.get("num_of_products", 1)),
                is_active_member=int(row.get("is_active_member", 1)),
            )})
        except (KeyError, ValueError, TypeError) as e:
            results.append({"input": row, "error": str(e)})
    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)