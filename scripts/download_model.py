import os
import mlflow
import torch
import boto3
from urllib.parse import urlparse

# Configuration
REGISTERED_MODEL_NAME = "style_transfer_photo2monet_cyclegan"
ALIAS = "prod"
ARTIFACT_NAME = "photo2monet_cyclegan.pt"
ARTIFACT_SUBPATH = f"models/{ARTIFACT_NAME}"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_USERNAME = os.getenv("MLFLOW_USERNAME")
MLFLOW_PASSWORD = os.getenv("MLFLOW_PASSWORD")

# Auth if needed
if MLFLOW_USERNAME and MLFLOW_PASSWORD:
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

# Resolve version from alias
version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS)
run_id = version.run_id
artifact_uri = f"runs:/{run_id}/models/{ARTIFACT_NAME}"

print(f"Trying MLflow artifact fetch from: {artifact_uri}")

try:
    local_path = mlflow.artifacts.download_artifacts(artifact_uri)
    print(f"Downloaded via MLflow to: {local_path}")

except Exception as e:
    print(f"MLflow download failed: {e}")
    print("Attempting direct S3 fallback...")

    # Step 1: Get full artifact URI for the run
    run = client.get_run(run_id)
    base_uri = run.info.artifact_uri  # e.g., s3://bucket-name/mlflow-artifacts/...
    print(f"Run artifact base URI: {base_uri}")

    # Step 2: Parse and build full path
    parsed = urlparse(base_uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")  # e.g., mlflow-artifacts/1/<run_id>/artifacts
    full_key = f"{prefix}/models/{ARTIFACT_NAME}"
    print(f"S3 URI: s3://{bucket}/{full_key}")

    # Step 3: Download using boto3
    session = boto3.session.Session()
    s3 = session.client(
        service_name="s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL")  # required for DO Spaces
    )

    local_path = f"./{ARTIFACT_NAME}"
    try:
        s3.download_file(bucket, full_key, local_path)
        print(f"Downloaded via S3 to: {local_path}")
    except Exception as s3e:
        print(f"ERROR: Could not download model from S3: {s3e}")
        raise