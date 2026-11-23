import os
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("data/customers.csv")

print("Dataset loaded:", df.shape)

# Features and target
X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]

# -----------------------------
# 2. Identify feature types
# -----------------------------
categorical_features = ["ContractType", "PaymentMethod"]

numeric_features = [
    "Age",
    "Tenure",
    "MonthlyCharges",
    "Complaints",
    "SupportCalls",
    "InternetUsageGB"
]

# -----------------------------
# 3. Preprocessing
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric", "passthrough", numeric_features)
    ]
)

# -----------------------------
# 4. Create ML pipeline
# -----------------------------
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        class_weight="balanced"
    ))
])

# -----------------------------
# 5. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 6. Train model
# -----------------------------
print("\nTraining model...")
model.fit(X_train, y_train)

# -----------------------------
# 7. Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 8. Evaluate model
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

cm = confusion_matrix(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy : {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall   : {recall:.2%}")
print(f"F1 Score : {f1:.2%}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# -----------------------------
# 9. Save trained model
# -----------------------------
os.makedirs("models", exist_ok=True)

import joblib

joblib.dump(model, "models/churn_model.pkl")

# -----------------------------
# 10. Save evaluation metrics
# -----------------------------
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist(),
    "test_samples": int(len(y_test)),
    "train_samples": int(len(X_train))
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nModel saved to models/churn_model.pkl")
print("Metrics saved to models/metrics.json")
print("\nTRAINING COMPLETE!")