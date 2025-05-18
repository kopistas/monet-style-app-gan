#!/usr/bin/env python3
"""
This script creates a production_model.json marker file from MLflow artifacts.
It helps integrate MLflow with the dynamic model loading system.

Usage:
  python scripts/create_production_marker.py --run-id <mlflow_run_id> \
      --artifact-path <path_to_model_in_mlflow> \
      --model-name <model_name> \
      --bucket <s3_bucket> \
      [--output production_model.json]
"""

import os
import json
import argparse
import logging
import sys
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def get_mlflow_model_path(run_id, artifact_path):
    """Get the S3 path for an MLflow model artifact"""
    try:
        import mlflow
        
        # Get MLflow tracking URI and artifact URI
        tracking_uri = mlflow.get_tracking_uri()
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        
        # Get the artifact URI for the run
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        artifact_uri = run.info.artifact_uri
        logger.info(f"Run artifact URI: {artifact_uri}")
        
        # Handle different MLflow storage backends
        if artifact_uri.startswith("s3://"):
            # S3 storage backend
            s3_path = os.path.join(artifact_uri, artifact_path)
            return s3_path.replace("s3://", "")
        else:
            # Local file storage or other backend
            local_path = os.path.join(mlflow.artifacts.download_artifacts(
                artifact_uri=os.path.join(artifact_uri, artifact_path)
            ))
            logger.info(f"Downloaded MLflow artifact to {local_path}")
            return local_path
            
    except ImportError:
        logger.error("MLflow is not installed. Please install it with 'pip install mlflow'")
        return None
    except Exception as e:
        logger.error(f"Error accessing MLflow artifact: {e}")
        return None

def create_production_marker(run_id, model_key, model_name, version=None):
    """Create a production marker file pointing to the model"""
    try:
        import mlflow
        
        # Get run info from MLflow if available
        try:
            run = mlflow.get_run(run_id)
            metrics = run.data.metrics
            tags = run.data.tags
        except Exception as e:
            logger.warning(f"Could not retrieve run details from MLflow: {e}")
            metrics = {}
            tags = {}
            
        # Extract relevant metrics for model info
        # Typical CycleGAN metrics
        selected_metrics = {
            k: v for k, v in metrics.items() 
            if any(term in k.lower() for term in ["loss", "score", "accuracy", "fid"])
        }
        
        # Create marker content
        marker = {
            "model_key": model_key,
            "version": version or datetime.now().strftime("%Y%m%d%H%M%S"),
            "promoted_at": datetime.now().isoformat(),
            "info": {
                "run_id": run_id,
                "model_name": model_name,
                "metrics": selected_metrics,
                "tags": {k: v for k, v in tags.items() if not k.startswith("mlflow.")}
            }
        }
        
        return marker
        
    except ImportError:
        logger.warning("MLflow not installed, creating basic marker")
        return {
            "model_key": model_key,
            "version": version or datetime.now().strftime("%Y%m%d%H%M%S"),
            "promoted_at": datetime.now().isoformat(),
            "info": {
                "run_id": run_id,
                "model_name": model_name
            }
        }

def upload_marker_to_s3(marker, bucket):
    """Upload the marker file to S3"""
    s3_endpoint = os.environ.get('S3_ENDPOINT')
    aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    
    try:
        # Create S3 client
        if s3_endpoint and aws_access_key and aws_secret_key:
            s3_client = boto3.client(
                's3',
                endpoint_url=s3_endpoint,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
        else:
            s3_client = boto3.client('s3')
        
        # Upload the marker
        marker_json = json.dumps(marker, indent=2)
        s3_client.put_object(
            Bucket=bucket,
            Key="production_model.json",
            Body=marker_json,
            ContentType="application/json"
        )
        
        logger.info(f"Production marker uploaded to s3://{bucket}/production_model.json")
        return True
    
    except Exception as e:
        logger.error(f"Error uploading marker to S3: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Create a production marker for an MLflow model")
    parser.add_argument("--run-id", required=True, help="MLflow run ID")
    parser.add_argument("--artifact-path", required=True, help="Path to model in MLflow artifacts")
    parser.add_argument("--model-name", required=True, help="Model name")
    parser.add_argument("--version", help="Model version (default: timestamp)")
    parser.add_argument("--bucket", help="S3 bucket to upload marker to")
    parser.add_argument("--output", default="production_model.json", help="Output file for marker")
    
    args = parser.parse_args()
    
    # Get model path from MLflow
    model_path = get_mlflow_model_path(args.run_id, args.artifact_path)
    if not model_path:
        logger.error("Failed to get model path from MLflow")
        return 1
    
    # Create marker
    logger.info(f"Creating production marker for model at {model_path}")
    marker = create_production_marker(args.run_id, model_path, args.model_name, args.version)
    
    # Save locally
    with open(args.output, "w") as f:
        json.dump(marker, f, indent=2)
    logger.info(f"Production marker saved to {args.output}")
    
    # Upload to S3 if bucket provided
    if args.bucket:
        if upload_marker_to_s3(marker, args.bucket):
            logger.info(f"Model {args.model_name} from run {args.run_id} set as production model")
        else:
            logger.error("Failed to upload marker to S3")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 