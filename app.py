
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained Decision Tree model
model = joblib.load("model.pkl")

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
        values = request.get_json() if request.is_json else request.form

        row = {
            feature: int(values[feature])
            for feature in FEATURES
        }

        if any(value not in (0, 1) for value in row.values()):
            raise ValueError("Symptoms must be 0 or 1.")

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
    app.run(host="0.0.0.0", port=5001, debug=False)
