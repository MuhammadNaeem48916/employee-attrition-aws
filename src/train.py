"""
Employee Attrition Training Script.
Saves model locally + uploads artifact to S3 (LocalStack).
"""
import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import boto3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report
)

# ── AWS / LocalStack config ──────────────────────────────
ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")
BUCKET   = "employee-attrition-mlops"
REGION   = "us-east-1"

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    region_name=REGION,
    aws_access_key_id="test",
    aws_secret_access_key="test"
)


def load_data(path="data/raw/attrition.csv"):
    df = pd.read_csv(path)
    X  = df.drop("attrition", axis=1)
    y  = df["attrition"]
    print(f"Loaded: {len(df)} rows | attrition rate: {y.mean():.1%}")
    return X, y


def get_model(model_type, params):
    if model_type == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            random_state=42
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 6),
            class_weight="balanced",
            random_state=42
        )
    elif model_type == "logistic":
        clf = LogisticRegression(
            C=params.get("C", 1.0),
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf)
    ])


def upload_to_s3(local_path, s3_key):
    """Upload file to S3 (LocalStack or real AWS)."""
    try:
        s3.upload_file(local_path, BUCKET, s3_key)
        print(f"S3 upload ✓  s3://{BUCKET}/{s3_key}")
    except Exception as e:
        print(f"S3 upload failed: {e}")


def train(model_type="gradient_boosting", params=None):
    if params is None:
        params = {}

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = get_model(model_type, params)
    model.fit(X_train, y_train)

    preds      = model.predict(X_test)
    preds_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_type": model_type,
        "params":     params,
        "accuracy":   round(accuracy_score(y_test, preds), 4),
        "f1_score":   round(f1_score(y_test, preds), 4),
        "precision":  round(precision_score(y_test, preds), 4),
        "recall":     round(recall_score(y_test, preds), 4),
        "roc_auc":    round(roc_auc_score(y_test, preds_prob), 4),
    }

    print("\n" + "=" * 52)
    print(f"  Model     : {model_type}")
    print(f"  Accuracy  : {metrics['accuracy']}")
    print(f"  Recall    : {metrics['recall']}  ← catch leavers")
    print(f"  Precision : {metrics['precision']}")
    print(f"  F1        : {metrics['f1_score']}")
    print(f"  ROC-AUC   : {metrics['roc_auc']}")
    print("=" * 52)
    print(classification_report(
        y_test, preds,
        target_names=["Stayed", "Left"]
    ))

    # save locally
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    model_path   = "models/model.pkl"
    metrics_path = "metrics/scores.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved  → {model_path}")
    print(f"Metrics saved → {metrics_path}")

    # upload to S3
    upload_to_s3(model_path,   f"models/{model_type}/v1/model.pkl")
    upload_to_s3(metrics_path, f"models/{model_type}/v1/metrics.json")

    return metrics


if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else \
                 "gradient_boosting"
    params = {}
    for arg in sys.argv[2:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            try: v = int(v)
            except ValueError:
                try: v = float(v)
                except ValueError: pass
            params[k] = v
    train(model_type, params)
