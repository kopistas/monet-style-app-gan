#!/usr/bin/env python3
import os
import sys
import json
import argparse
import boto3
from botocore.exceptions import ClientError
import mlflow
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def get_s3_client():
    """Create and configure S3 client"""
    s3_endpoint = os.getenv('S3_ENDPOINT')
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if s3_endpoint and aws_access_key and aws_secret_key:
        logger.info(f"Creating S3 client with endpoint: {s3_endpoint}")
        return boto3.client(
            's3',
            endpoint_url=s3_endpoint,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
    elif aws_access_key and aws_secret_key:
        logger.info("Creating S3 client with AWS credentials")
        return boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
    else:
        logger.error("No S3 credentials provided")
        return None

def upload_model_to_s3(model_path, bucket_name, model_key=None):
    """Upload model file to S3 bucket with progress tracking"""
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
    
    s3_client = get_s3_client()
    if not s3_client:
        return None
    
    # Generate model key if not provided (use filename)
    if not model_key:
        model_key = f"models/{os.path.basename(model_path)}"
    
    try:
        # Get file size for progress tracking
        file_size = os.path.getsize(model_path)
        logger.info(f"Uploading model {model_path} ({file_size/1024/1024:.2f} MB) to s3://{bucket_name}/{model_key}")
        
        # Define callback to track upload progress
        uploaded_bytes = 0
        
        def progress_callback(bytes_amount):
            nonlocal uploaded_bytes
            uploaded_bytes += bytes_amount
            progress = int((uploaded_bytes / file_size) * 100)
            sys.stdout.write(f"\rUpload progress: {progress}% ({uploaded_bytes/1024/1024:.2f} MB of {file_size/1024/1024:.2f} MB)")
            sys.stdout.flush()
        
        # Upload file with progress tracking
        s3_client.upload_file(
            model_path, 
            bucket_name, 
            model_key,
            Callback=progress_callback
        )
        
        print()  # newline after progress
        logger.info(f"Upload completed: s3://{bucket_name}/{model_key}")
        return model_key
    
    except ClientError as e:
        logger.error(f"Error uploading model to S3: {e}")
        return None

def update_production_marker(bucket_name, model_key, version, model_info=None):
    """Update the production marker file to point to the new model"""
    s3_client = get_s3_client()
    if not s3_client:
        return False
    
    production_marker_key = "production_model.json"
    
    # Prepare model information
    if model_info is None:
        model_info = {}
    
    production_info = {
        "model_key": model_key,
        "version": version,
        "promoted_at": datetime.now().isoformat(),
        "info": model_info
    }
    
    try:
        # Convert to JSON and upload
        json_data = json.dumps(production_info, indent=2)
        s3_client.put_object(
            Bucket=bucket_name,
            Key=production_marker_key,
            Body=json_data,
            ContentType="application/json"
        )
        
        logger.info(f"Updated production marker to point to model: {model_key}")
        return True
    
    except ClientError as e:
        logger.error(f"Error updating production marker: {e}")
        return False

def get_mlflow_model_info(run_id=None, model_name=None):
    """Get model information from MLflow if available"""
    if not run_id and not model_name:
        logger.warning("No MLflow run_id or model_name provided")
        return None
    
    try:
        # Try to import mlflow
        import mlflow
        
        model_info = {}
        
        if run_id:
            # Get run info
            run = mlflow.get_run(run_id)
            model_info["run_id"] = run_id
            model_info["metrics"] = run.data.metrics
            model_info["tags"] = run.data.tags
        
        if model_name:
            # Get latest model version
            try:
                latest_version = mlflow.tracking.MlflowClient().get_latest_versions(model_name, stages=["None"])
                if latest_version:
                    model_info["model_name"] = model_name
                    model_info["model_version"] = latest_version[0].version
            except Exception as e:
                logger.warning(f"Error getting model version: {e}")
        
        return model_info
    
    except ImportError:
        logger.warning("MLflow not installed, skipping model info")
        return None
    except Exception as e:
        logger.warning(f"Error getting MLflow model info: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Upload and promote a model to production in S3")
    parser.add_argument("model_path", help="Path to the model file to upload")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--model-key", help="S3 key to use for the model (default: models/filename)")
    parser.add_argument("--version", help="Model version (default: timestamp)")
    parser.add_argument("--run-id", help="MLflow run ID for metadata")
    parser.add_argument("--model-name", help="MLflow model name for metadata")
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return 1
    
    # Set version if not provided
    if not args.version:
        args.version = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Get MLflow metadata if possible
    model_info = get_mlflow_model_info(args.run_id, args.model_name)
    
    # Upload model to S3
    model_key = upload_model_to_s3(args.model_path, args.bucket, args.model_key)
    if not model_key:
        return 1
    
    # Update production marker
    if not update_production_marker(args.bucket, model_key, args.version, model_info):
        return 1
    
    logger.info("=== Model successfully promoted to production ===")
    logger.info(f"Model: {model_key}")
    logger.info(f"Version: {args.version}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 