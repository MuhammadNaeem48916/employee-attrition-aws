"""
Employee Attrition Prediction API.
Logs metrics to CloudWatch on every prediction.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import boto3
import time
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Employee Attrition Prediction API",
    version="1.0.0"
)

MODEL_PATH  = os.getenv("MODEL_PATH", "models/model.pkl")
CW_ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")
model       = None
model_ready = False

cw = boto3.client(
    "cloudwatch",
    endpoint_url=CW_ENDPOINT,
    region_name="us-east-1",
    aws_access_key_id="test",
    aws_secret_access_key="test"
)


@app.on_event("startup")
def load_model():
    global model, model_ready
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        model_ready = True
        logger.info(f"Model loaded from {MODEL_PATH} ✓")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


def log_to_cloudwatch(prediction, latency_ms, confidence):
    """Log prediction metrics to CloudWatch."""
    try:
        from datetime import datetime
        cw.put_metric_data(
            Namespace="EmployeeAttrition/Model",
            MetricData=[
                {
                    "MetricName": "PredictionLatency",
                    "Value":      latency_ms,
                    "Unit":       "Milliseconds",
                    "Timestamp":  datetime.utcnow()
                },
                {
                    "MetricName": "PredictionCount",
                    "Value":      1,
                    "Unit":       "Count",
                    "Dimensions": [{
                        "Name":  "Result",
                        "Value": "attrition" if prediction else "stay"
                    }],
                    "Timestamp": datetime.utcnow()
                }
            ]
        )
    except Exception as e:
        logger.warning(f"CloudWatch logging failed: {e}")


# ── Health endpoints ─────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "alive"}


@app.get("/ready")
def ready():
    if not model_ready:
        raise HTTPException(503, "Model not loaded")
    return {"status": "ready"}


@app.get("/")
def root():
    return {
        "service": "Employee Attrition API",
        "status":  "running",
        "docs":    "/docs"
    }


# ── Schema ───────────────────────────────────────────────
class EmployeeFeatures(BaseModel):
    age:                        float
    monthly_income:             float
    years_at_company:           float
    overtime:                   float
    job_satisfaction:           float
    work_life_balance:          float
    distance_from_home:         float
    num_companies_worked:       float
    training_times_last_year:   float
    performance_rating:         float

    class Config:
        json_schema_extra = {"example": {
            "age": 26,
            "monthly_income": 3200,
            "years_at_company": 1,
            "overtime": 1,
            "job_satisfaction": 2,
            "work_life_balance": 2,
            "distance_from_home": 22,
            "num_companies_worked": 5,
            "training_times_last_year": 1,
            "performance_rating": 2
        }}


# ── Prediction ───────────────────────────────────────────
@app.post("/predict")
def predict(employee: EmployeeFeatures):
    if not model_ready:
        raise HTTPException(503, "Model not ready")

    import pandas as pd
    start = time.time()

    df         = pd.DataFrame([employee.dict()])
    pred       = int(model.predict(df)[0])
    prob       = model.predict_proba(df)[0]
    latency_ms = (time.time() - start) * 1000
    confidence = round(float(max(prob)), 4)

    attrition_prob = round(float(prob[1]), 4)
    risk = "HIGH"   if attrition_prob >= 0.65 else \
           "MEDIUM" if attrition_prob >= 0.35 else "LOW"

    # log to CloudWatch
    log_to_cloudwatch(pred, latency_ms, confidence)

    return {
        "attrition_prediction":  pred,
        "attrition_label":       "Will Leave" if pred else "Will Stay",
        "attrition_probability": attrition_prob,
        "stay_probability":      round(float(prob[0]), 4),
        "risk_level":            risk,
        "latency_ms":            round(latency_ms, 2)
    }


@app.get("/model-info")
def model_info():
    return {
        "model_path":  MODEL_PATH,
        "model_ready": model_ready
    }
