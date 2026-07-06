"""
S3 operations using LocalStack.
Identical API to real AWS S3 — change endpoint_url for production.
"""
import boto3
import os

ENDPOINT = "http://localhost:4566"
BUCKET   = "employee-attrition-mlops"
REGION   = "us-east-1"

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    region_name=REGION,
    aws_access_key_id="test",
    aws_secret_access_key="test"
)


def create_bucket():
    try:
        s3.create_bucket(Bucket=BUCKET)
        # enable versioning — every overwrite keeps old version
        s3.put_bucket_versioning(
            Bucket=BUCKET,
            VersioningConfiguration={"Status": "Enabled"}
        )
        print(f"Bucket created : s3://{BUCKET} ✓")
        print(f"Versioning     : enabled ✓")
    except Exception as e:
        print(f"Bucket note    : {e}")


def upload_dataset():
    s3.upload_file(
        "data/raw/attrition.csv",
        BUCKET,
        "data/raw/attrition.csv"
    )
    print(f"Dataset uploaded → s3://{BUCKET}/data/raw/attrition.csv ✓")


def list_bucket():
    response = s3.list_objects_v2(Bucket=BUCKET)
    objects  = response.get("Contents", [])
    print(f"\nFiles in s3://{BUCKET}/ ({len(objects)} objects):")
    for obj in objects:
        size_kb = obj["Size"] / 1024
        print(f"  {obj['Key']:55s} {size_kb:.1f} KB")


def download_model(model_type="gradient_boosting", version="v1"):
    key        = f"models/{model_type}/{version}/model.pkl"
    local_path = f"models/model_from_s3.pkl"
    s3.download_file(BUCKET, key, local_path)
    print(f"Model downloaded → {local_path} ✓")
    return local_path


if __name__ == "__main__":
    print("=" * 50)
    print("  S3 Operations Demo (LocalStack)")
    print("=" * 50)
    create_bucket()
    upload_dataset()
    list_bucket()
