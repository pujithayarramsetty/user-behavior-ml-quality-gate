import pandas as pd
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 1. Load dataset
data = pd.read_csv("user_behavior_dataset.csv")

print("Dataset shape:", data.shape)


# 2. Separate features and target
X = data.drop(columns=["User Behavior Class"])
y = data["User Behavior Class"]


# 3. Define categorical columns
categorical_columns = [
    "Device Model",
    "Operating System",
    "Gender"
]


# 4. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# 5. Create ML pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)


# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 7. Train model
model.fit(X_train, y_train)


# 8. Prediction
y_pred = model.predict(X_test)


# 9. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy, 4))


# 10. Save model
with open("user_behavior_model.pkl", "wb") as file:
    pickle.dump(model, file)


# 11. Save evaluation metrics
metrics = {
    "accuracy": float(accuracy),
    "training_records": len(X_train),
    "testing_records": len(X_test)
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)


print("Model saved successfully.")
print("Metrics saved successfully.")
