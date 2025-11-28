import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import pickle

# Load dataset
df = pd.read_csv("diabetes_MLOps.csv")

# Pisahkan fitur dan label (Outcome)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pastikan folder models ada
os.makedirs("models", exist_ok=True)

# Set MLflow experiment
mlflow.set_experiment("diabetes_classification")

with mlflow.start_run():

    # Hyperparameters
    n_estimators = 200
    max_depth = 5

    # Model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Logging ke MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)

    # Logging model
    mlflow.sklearn.log_model(model, "model")

    # Save model ke models/
    with open("models/diabetes_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Training selesai.")
    print("Akurasi:", acc)
