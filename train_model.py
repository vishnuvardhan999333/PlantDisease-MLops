
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Demonstration dataset: 0 = absent, 1 = present
data = {
    "leaf_spots": [
        1, 1, 1, 1, 0, 0, 0, 0,
        1, 1, 0, 0, 1, 0, 1, 0,
        1, 1, 0, 0
    ],
    "yellowing": [
        0, 0, 1, 0, 1, 1, 0, 0,
        1, 0, 1, 1, 0, 0, 1, 1,
        0, 1, 0, 1
    ],
    "leaf_curl": [
        0, 0, 0, 1, 0, 0, 1, 0,
        0, 1, 0, 1, 1, 1, 0, 0,
        0, 0, 1, 1
    ],
    "wilting": [
        0, 0, 0, 0, 1, 1, 1, 0,
        0, 0, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0
    ],
    "disease": [
        "Leaf Spot", "Leaf Spot", "Leaf Spot", "Leaf Spot",
        "Powdery Mildew", "Powdery Mildew", "Powdery Mildew",
        "Healthy", "Leaf Spot", "Leaf Spot",
        "Powdery Mildew", "Powdery Mildew", "Leaf Spot",
        "Powdery Mildew", "Leaf Spot", "Powdery Mildew",
        "Leaf Spot", "Leaf Spot", "Powdery Mildew",
        "Powdery Mildew"
    ]
}

df = pd.DataFrame(data)

features = ["leaf_spots", "yellowing", "leaf_curl", "wilting"]
X = df[features]
y = df["disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n")
print(classification_report(y_test, predictions, zero_division=0))

model_path = Path(__file__).resolve().parent / "model.pkl"
joblib.dump(model, model_path)
print("\nModel saved successfully as model.pkl")
print("Input features:", features)
