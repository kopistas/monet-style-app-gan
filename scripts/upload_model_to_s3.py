#!/usr/bin/env python3
"""
Upload a model file to S3-compatible storage (DigitalOcean Spaces)

This script uploads the specified model file to a DigitalOcean Spaces bucket.
It uses environment variables for configuration:

Required environment variables:
- AWS_ACCESS_KEY_ID: S3 access key
- AWS_SECRET_ACCESS_KEY: S3 secret key
- S3_ENDPOINT: S3 endpoint URL (https://region.digitaloceanspaces.com)
- BUCKET_NAME: S3 bucket name
- AWS_REGION: S3 region

Optional environment variables:
- MODEL_PATH: Path to the model file (default: photo2monet_cyclegan_mark4.pt)
- MODEL_DEST_PATH: Destination path in S3 (default: same as MODEL_PATH)
"""

import boto3
import os
import sys
from botocore.client import Config

def upload_model_to_s3():
    """Upload the model file to S3 bucket"""
    # Get required environment variables
    endpoint_url = os.environ.get('S3_ENDPOINT')
    if not endpoint_url:
        print("ERROR: S3_ENDPOINT environment variable is required")
        sys.exit(1)
        
    bucket = os.environ.get('BUCKET_NAME')
    if not bucket:
        print("ERROR: BUCKET_NAME environment variable is required")
        sys.exit(1)
        
    aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    if not aws_access_key:
        print("ERROR: AWS_ACCESS_KEY_ID environment variable is required")
        sys.exit(1)
        
    aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    if not aws_secret_key:
        print("ERROR: AWS_SECRET_ACCESS_KEY environment variable is required")
        sys.exit(1)

    aws_region = os.environ.get('AWS_REGION')
    if not aws_region:
        print("ERROR: AWS_REGION environment variable is required")
        sys.exit(1)
        
    # Get optional environment variables
    model_path = os.environ.get('MODEL_PATH', 'photo2monet_cyclegan_mark4.pt')
    model_dest_path = os.environ.get('MODEL_DEST_PATH', model_path)
    
    print(f"S3 Configuration:")
    print(f"  Endpoint URL: {endpoint_url}")
    print(f"  Bucket: {bucket}")
    print(f"  Region: {aws_region}")
    print(f"  Model path: {model_path}")
    print(f"  Destination path: {model_dest_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file '{model_path}' not found")
        print(f"Current directory: {os.getcwd()}")
        print(f"Directory contents: {os.listdir('.')}")
        sys.exit(1)

    try:
        # Configure and create S3 client
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            config=Config(signature_version='s3v4'),
            region_name=aws_region
        )
        
        # Upload the model
        print(f"Uploading model {model_path} to bucket {bucket} as {model_dest_path}")
        s3_client.upload_file(model_path, bucket, model_dest_path)
        print(f"Model successfully uploaded to {bucket}/{model_dest_path}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to upload model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    upload_model_to_s3() 