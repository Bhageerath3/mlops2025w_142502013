# src/train.py
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from mlflow.models.signature import infer_signature

import matplotlib
matplotlib.use("Agg")           # <--- ADD THESE TWO LINES
import matplotlib.pyplot as plt
import seaborn as sns
import os



def load_and_explore_data():
    print("Loading Iris dataset...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")
    df = X.copy()
    df["species"] = y
    df["species_name"] = df["species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Classes: {iris.target_names}")
    print(f"Class distribution:\n{df['species_name'].value_counts()}")
    return X, y, df


def preprocess_data(X, y, test_size=0.2, random_state=42):
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    print(f"Training set size: {X_train_scaled.shape[0]}")
    print(f"Test set size: {X_test_scaled.shape[0]}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
    }


def _start_run_safely(run_name):
    active = mlflow.active_run()
    if active is None:
        return mlflow.start_run(run_name=run_name)
    else:
        return mlflow.start_run(nested=True, run_name=run_name)


def _log_and_download_model(sk_model, X_test, y_pred_test, model_name="model"):
    """
    Log the model to MLflow (within an active run), then download the artifact
    to a local path using mlflow.artifacts.download_artifacts(...) and return:
      (artifact_uri, local_model_path)
    local_model_path is a local directory path (no file:// prefix).
    """
    signature = infer_signature(X_test, y_pred_test)
    mlflow.sklearn.log_model(
        sk_model=sk_model,
        name=model_name,
        signature=signature,
        input_example=X_test.head(3),
    )

    run_id = mlflow.active_run().info.run_id
    # download the 'model' artifact from the run to a local path
    try:
        local_model_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=model_name)
    except Exception as e:
        # fallback: try downloading by artifact_uri
        artifact_uri = mlflow.get_artifact_uri(model_name)
        local_model_path = None
        try:
            local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
        except Exception:
            raise RuntimeError(f"Failed to download model artifact (run_id={run_id}, artifact_path={model_name}): {e}")

    if not os.path.exists(local_model_path):
        raise RuntimeError(f"Downloaded model path does not exist: {local_model_path}")

    artifact_uri = mlflow.get_artifact_uri(model_name)
    return artifact_uri, local_model_path


def train_random_forest(X_train, y_train, X_test, y_test, **params):
    with _start_run_safely(run_name="Random Forest Classifier"):
        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_metrics = calculate_metrics(y_train, y_pred_train)
        test_metrics = calculate_metrics(y_test, y_pred_test)

        for metric, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric}", value)
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)

        # confusion matrix plot
        cm = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix - Random Forest")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.savefig("confusion_matrix_rf.png")
        mlflow.log_artifact("confusion_matrix_rf.png")
        plt.close()

        # log model and download local copy
        artifact_uri, local_model_path = _log_and_download_model(model, X_test, y_pred_test, model_name="model")

        run_id = mlflow.active_run().info.run_id
        mlflow.set_tag("model_family", "tree_based")
        mlflow.set_tag("dataset", "iris")

        print(f"Random Forest - Test Accuracy: {test_metrics['accuracy']:.4f} (run_id={run_id}, artifact_uri={artifact_uri}, local_path={local_model_path})")
        return model, test_metrics["accuracy"], run_id, artifact_uri, local_model_path


def train_logistic_regression(X_train, y_train, X_test, y_test, **params):
    with _start_run_safely(run_name="Logistic Regression"):
        mlflow.log_params(params)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_metrics = calculate_metrics(y_train, y_pred_train)
        test_metrics = calculate_metrics(y_test, y_pred_test)

        for metric, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric}", value)
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)

        cm = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
        plt.title("Confusion Matrix - Logistic Regression")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.savefig("confusion_matrix_lr.png")
        mlflow.log_artifact("confusion_matrix_lr.png")
        plt.close()

        artifact_uri, local_model_path = _log_and_download_model(model, X_test, y_pred_test, model_name="model")
        run_id = mlflow.active_run().info.run_id
        mlflow.set_tag("model_family", "linear")
        mlflow.set_tag("dataset", "iris")

        print(f"Logistic Regression - Test Accuracy: {test_metrics['accuracy']:.4f} (run_id={run_id}, artifact_uri={artifact_uri}, local_path={local_model_path})")
        return model, test_metrics["accuracy"], run_id, artifact_uri, local_model_path


def hyperparameter_tuning():
    X, y, df = load_and_explore_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    mlflow.set_experiment("Iris Classification Hyperparameter Tuning")

    best_accuracy = -1.0
    best_model = None
    best_run_id = None
    best_artifact_uri = None
    best_local_model_path = None

    with mlflow.start_run(run_name="Hyperparameter Tuning Session"):
        rf_params = [
            {"n_estimators": 50, "max_depth": 3, "random_state": 42},
            {"n_estimators": 100, "max_depth": 5, "random_state": 42},
            {"n_estimators": 200, "max_depth": 7, "random_state": 42},
            {"n_estimators": 100, "max_depth": None, "random_state": 42},
        ]
        for params in rf_params:
            with mlflow.start_run(nested=True, run_name=f"RF_estimators_{params['n_estimators']}_depth_{params['max_depth']}"):
                model, acc, run_id, artifact_uri, local_path = train_random_forest(X_train, y_train, X_test, y_test, **params)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model = model
                    best_run_id = run_id
                    best_artifact_uri = artifact_uri
                    best_local_model_path = local_path

        lr_params = [
            {"C": 0.1, "random_state": 42, "max_iter": 1000},
            {"C": 1.0, "random_state": 42, "max_iter": 1000},
            {"C": 10.0, "random_state": 42, "max_iter": 1000},
        ]
        for params in lr_params:
            with mlflow.start_run(nested=True, run_name=f"LR_C_{params['C']}"):
                model, acc, run_id, artifact_uri, local_path = train_logistic_regression(X_train, y_train, X_test, y_test, **params)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model = model
                    best_run_id = run_id
                    best_artifact_uri = artifact_uri
                    best_local_model_path = local_path

        mlflow.log_metric("best_accuracy", best_accuracy)
        mlflow.log_param("best_run_id", best_run_id)
        mlflow.log_param("best_artifact_uri", best_artifact_uri)
        mlflow.log_param("best_local_model_path", best_local_model_path)
        mlflow.set_tag("status", "completed")

        print(f"\nBest model accuracy: {best_accuracy:.4f}")
        print(f"Best run ID: {best_run_id}")
        print(f"Best artifact URI: {best_artifact_uri}")
        print(f"Best local model path: {best_local_model_path}")

    return best_model, best_run_id, best_artifact_uri, best_local_model_path


def model_registry_example(best_run_id, best_artifact_uri, best_local_model_path):
    client = mlflow.tracking.MlflowClient()
    model_name = "iris-classifier"

    print(f"Registering model from local path: {best_local_model_path} (run {best_run_id})")

    # Use the local path to register (add file:// prefix so MLflow knows it's a local file)
    model_uri_for_registration = (
        best_local_model_path if best_local_model_path.startswith("file:") else f"file://{best_local_model_path}"
    )

    try:
        model_version = mlflow.register_model(model_uri=model_uri_for_registration, name=model_name)
        print(f"Model registered: {model_name}, Version: {model_version.version}")
    except Exception as e:
        raise RuntimeError(f"Failed to register model from {model_uri_for_registration}: {e}") from e

    # Best-effort metadata updates (catch exceptions)
    try:
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description="Best performing model from hyperparameter tuning session",
        )
    except Exception as e:
        print("Warning: update_model_version() failed (likely YAML/serialization issue). Continuing. Error:", e)

    try:
        client.set_model_version_tag(name=model_name, version=model_version.version, key="stage", value="production_candidate")
    except Exception as e:
        print("Warning: set_model_version_tag() failed:", e)

    try:
        client.transition_model_version_stage(name=model_name, version=model_version.version, stage="Production")
    except Exception as e:
        print("Warning: transition_model_version_stage() failed (may be fine on local file store):", e)

    return model_name, model_version.version


def load_and_predict(model_name, model_version, fallback_local_path=None):
    tried = []
    model = None

    # First try registry by explicit version
    if model_name and model_version:
        registry_uri = f"models:/{model_name}/{model_version}"
        tried.append(registry_uri)
        print(f"Trying to load from registry URI: {registry_uri}")
        try:
            model = mlflow.sklearn.load_model(registry_uri)
            print(f"Loaded model from {registry_uri}")
        except Exception as e:
            print(f"Failed to load from registry URI {registry_uri}: {e}")
            model = None

    # Fallback to local downloaded path
    if model is None and fallback_local_path:
        tried.append(fallback_local_path)
        print(f"Trying to load from local path: {fallback_local_path}")
        try:
            model = mlflow.sklearn.load_model(fallback_local_path)
            print(f"Loaded model from local path {fallback_local_path}")
        except Exception as e:
            print(f"Failed to load from local path {fallback_local_path}: {e}")
            model = None

    if model is None:
        raise RuntimeError(f"Unable to load model. Tried: {tried}")

    X, y, df = load_and_explore_data()
    X_sample = X.head(5)
    predictions = model.predict(X_sample)

    results = pd.DataFrame({
        "sepal_length": X_sample["sepal length (cm)"],
        "sepal_width": X_sample["sepal width (cm)"],
        "petal_length": X_sample["petal length (cm)"],
        "petal_width": X_sample["petal width (cm)"],
        "predicted_class": predictions,
        "predicted_species": [["setosa", "versicolor", "virginica"][p] for p in predictions],
    })

    print("\nPrediction Results:")
    print(results.to_string(index=False))
    return results


def main():
    print("=== MLflow Iris Classification Pipeline ===\n")
    best_model, best_run_id, best_artifact_uri, best_local_model_path = hyperparameter_tuning()

    if best_local_model_path is None:
        raise RuntimeError("No local model path captured. Check training logs/artifacts.")

    model_name, version = model_registry_example(best_run_id, best_artifact_uri, best_local_model_path)

    results = load_and_predict(model_name, version, fallback_local_path=best_local_model_path)

    print(f"\n=== Pipeline Completed Successfully ===")
    print(f"Best model registered as: {model_name} (Version: {version})")
    print("MLflow UI: http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
