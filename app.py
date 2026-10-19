
from pathlib import Path

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "model.pkl")

FEATURES = [
    "leaf_spots",
    "yellowing",
    "leaf_curl",
    "wilting"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "GET":
        return jsonify({
            "message": "Plant Disease Prediction API",
            "features": FEATURES,
            "values": "0 = absent, 1 = present"
        })

    try:
        values = request.get_json(silent=True) if request.is_json else request.form
        if not isinstance(values, dict) and not request.is_json:
            values = request.form
        if not hasattr(values, "get"):
            raise ValueError("Expected symptom values.")

        row = {}
        for feature in FEATURES:
            value = values.get(feature)
            # HTML forms submit strings; JSON clients should send integer 0 or 1.
            is_binary_integer = type(value) is int and value in (0, 1)
            is_binary_form_value = isinstance(value, str) and value in ("0", "1")
            if not (is_binary_integer or is_binary_form_value):
                raise ValueError("Symptoms must be 0 or 1.")
            row[feature] = int(value)

        input_data = pd.DataFrame([row], columns=FEATURES)
        result = model.predict(input_data)[0]

        if request.is_json:
            return jsonify({
                "prediction": str(result),
                "input": row
            })

        return render_template(
            "index.html",
            result=result,
            inputs=row
        )

    except (ValueError, TypeError, KeyError):
        message = "Enter all four symptoms as 0 or 1."

        if request.is_json:
            return jsonify({"error": message}), 400

        return render_template("index.html", error=message), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
