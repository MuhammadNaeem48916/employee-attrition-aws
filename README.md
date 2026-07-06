# Employee Attrition Prediction — AWS MLOps Pipeline

> Predicts employee attrition risk using a full AWS-simulated
> MLOps pipeline. Built as Project 6 of a structured MLOps
> learning path.

---

## Problem Statement

Employee attrition costs companies 6–9 months of an employee's
salary in recruiting and training costs. Predicting who is at
risk of leaving lets HR teams intervene proactively.

**Challenge:** Only ~16% of employees leave — imbalanced dataset.
Recall matters most — missing a flight risk is costly.

---

## AWS Architecture (LocalStack simulation)

```
attrition.csv (DVC)
      ↓
S3 Bucket (versioned)
      ↓
Training Script (boto3)
      ↓
Model Artifact → S3
      ↓
FastAPI Serving Container
      ↓
CloudWatch (latency + predictions logged)
```

---

## Services Used

| Service | Purpose |
|---|---|
| S3 | Dataset + model artifact storage with versioning |
| CloudWatch | Prediction latency alarm + custom metrics |
| boto3 | Programmatic access from Python |
| LocalStack | Local AWS simulation (no cloud costs) |
| DVC | Data versioning |
| FastAPI | REST prediction API |
| Docker | Containerized serving |

---

## Quickstart

```bash
# start LocalStack
docker run -d --name localstack -p 4566:4566 \
  -e SERVICES=s3,cloudwatch localstack/localstack

# install dependencies
pip install -r requirements.txt

# create dataset
python src/create_dataset.py

# run tests
pytest tests/test_pipeline.py -v

# upload to S3
python aws/s3_operations.py

# train (saves to local + S3)
python src/train.py gradient_boosting n_estimators=100

# run CloudWatch monitoring
python aws/cloudwatch_monitoring.py

# serve API
docker build -f Dockerfile.serve -t employee-attrition-serving:v1 .
docker run -p 8000:8000 -v $(pwd)/models:/app/models \
  employee-attrition-serving:v1
```

---

## API Usage

```bash
# high risk employee
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 26, "monthly_income": 3200,
    "years_at_company": 1, "overtime": 1,
    "job_satisfaction": 1, "work_life_balance": 2,
    "distance_from_home": 22, "num_companies_worked": 5,
    "training_times_last_year": 1, "performance_rating": 2
  }'

# response
{
  "attrition_label": "Will Leave",
  "attrition_probability": 0.84,
  "risk_level": "HIGH",
  "latency_ms": 12.4
}
```

---

## Key concepts demonstrated

| Concept | Implementation |
|---|---|
| Object storage | S3 bucket with versioning enabled |
| Artifact management | Model uploaded to S3 after training |
| Custom metrics | Latency + prediction count in CloudWatch |
| Alarm | Fires when latency > 500ms |
| Data versioning | DVC tracks attrition.csv by hash |
| Unit testing | 8 tests covering data quality + model |
| Containerization | Docker serving image |

---

## Moving to real AWS

Change one line per script:
```python
# LocalStack (current)
endpoint_url="http://localhost:4566"

# Real AWS (production) — remove this line entirely
# boto3 automatically uses real AWS endpoints
```

---

## Skills demonstrated

`AWS` `S3` `CloudWatch` `boto3` `LocalStack` `MLOps`
`Docker` `FastAPI` `DVC` `Python` `Scikit-learn`

---

*Project 6 of MLOps → LLMOps → AgentOps learning path.*
*Previous: [Churn MLOps](https://github.com/MuhammadNaeem48916/churn-mlops)*
*Next: Skill 7 — Feature Stores*
