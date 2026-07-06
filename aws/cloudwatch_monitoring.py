"""
CloudWatch monitoring for ML serving.
Logs prediction metrics and sets up latency alarm.
"""
import boto3
import random
import time
import json
from datetime import datetime, timedelta

ENDPOINT  = "http://localhost:4566"
REGION    = "us-east-1"
NAMESPACE = "EmployeeAttrition/Model"

cw = boto3.client(
    "cloudwatch",
    endpoint_url=ENDPOINT,
    region_name=REGION,
    aws_access_key_id="test",
    aws_secret_access_key="test"
)


def setup_alarm():
    """Alert when prediction latency exceeds 500ms."""
    cw.put_metric_alarm(
        AlarmName="HighPredictionLatency",
        AlarmDescription="Fires when avg latency > 500ms",
        MetricName="PredictionLatency",
        Namespace=NAMESPACE,
        Statistic="Average",
        Period=60,
        EvaluationPeriods=2,
        Threshold=500.0,
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching"
    )
    print("Alarm created  : HighPredictionLatency (>500ms) ✓")


def log_prediction(prediction, latency_ms, confidence):
    """
    Log a single prediction to CloudWatch.
    Call this from serve.py on every request.
    """
    cw.put_metric_data(
        Namespace=NAMESPACE,
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
            },
            {
                "MetricName": "ModelConfidence",
                "Value":      confidence,
                "Unit":       "None",
                "Timestamp":  datetime.utcnow()
            }
        ]
    )


def simulate_traffic(n=20):
    """Simulate n prediction requests being logged."""
    print(f"\nSimulating {n} prediction requests...")
    attrition_count = 0

    for i in range(n):
        latency    = random.uniform(25, 200)
        prediction = random.choice([0, 0, 0, 0, 1])  # ~20% attrition
        confidence = random.uniform(0.6, 0.95)

        log_prediction(prediction, latency, confidence)
        if prediction:
            attrition_count += 1

        print(f"  Request {i+1:2d}: "
              f"{'ATTRITION' if prediction else 'STAY':9s} | "
              f"latency: {latency:5.1f}ms | "
              f"confidence: {confidence:.2f}")

    print(f"\nSummary: {attrition_count}/{n} predicted as attrition")


def get_statistics():
    """Read back CloudWatch metrics."""
    response = cw.get_metric_statistics(
        Namespace=NAMESPACE,
        MetricName="PredictionLatency",
        StartTime=datetime.utcnow() - timedelta(hours=1),
        EndTime=datetime.utcnow(),
        Period=3600,
        Statistics=["Average", "Maximum", "Minimum", "SampleCount"]
    )

    points = response.get("Datapoints", [])
    if points:
        p = points[0]
        print(f"\nCloudWatch Statistics (last 1 hour):")
        print(f"  Requests : {int(p['SampleCount'])}")
        print(f"  Avg      : {p['Average']:.1f}ms")
        print(f"  Max      : {p['Maximum']:.1f}ms")
        print(f"  Min      : {p['Minimum']:.1f}ms")
    else:
        print("\nNo datapoints yet (metrics may take a moment)")


def list_alarms():
    """Show all CloudWatch alarms."""
    response = cw.describe_alarms()
    alarms   = response.get("MetricAlarms", [])
    print(f"\nCloudWatch Alarms ({len(alarms)}):")
    for alarm in alarms:
        print(f"  {alarm['AlarmName']:30s} "
              f"threshold: {alarm['Threshold']}"
              f"{alarm['Unit'] if alarm.get('Unit') else ''}")


if __name__ == "__main__":
    print("=" * 52)
    print("  CloudWatch Monitoring Demo (LocalStack)")
    print("=" * 52)
    setup_alarm()
    simulate_traffic(n=20)
    get_statistics()
    list_alarms()
    print("\nAll CloudWatch operations successful ✓")
