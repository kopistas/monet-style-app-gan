#!/bin/bash
set -e

# Configure AWS credentials for S3 access
if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
  echo "Configuring S3 credentials..."
  mkdir -p ~/.aws
  cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
EOF
  
  cat > ~/.aws/config << EOF
[default]
region = $AWS_REGION
endpoint_url = $S3_ENDPOINT
EOF

  echo "S3 configuration:"
  echo "S3_ENDPOINT: $S3_ENDPOINT"
  echo "AWS_REGION: $AWS_REGION"
  echo "S3_BUCKET: $S3_BUCKET"
fi

# Download the model - prioritize promoted model if specified, otherwise use default
MODEL_TO_DOWNLOAD=${PROMOTED_MODEL_NAME:-$DEFAULT_MODEL_NAME}
MODEL_PATH="/app/models/photo2monet_cyclegan_mark4.pt"

if [ -n "$MODEL_TO_DOWNLOAD" ] && [ -n "$S3_BUCKET" ]; then
  echo "Downloading model $MODEL_TO_DOWNLOAD from S3 bucket $S3_BUCKET..."
  python -c "
import boto3
import os
import sys
from botocore.client import Config
from botocore.exceptions import ClientError

try:
    print('Creating S3 client with:')
    print(f'  endpoint_url: {os.environ.get(\"S3_ENDPOINT\")}')
    print(f'  region: {os.environ.get(\"AWS_REGION\")}')
    print(f'  bucket: {os.environ.get(\"S3_BUCKET\")}')
    
    # Configure S3 client
    s3_client = boto3.client(
        's3',
        endpoint_url=os.environ.get('S3_ENDPOINT'),
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        config=Config(signature_version='s3v4'),
        region_name=os.environ.get('AWS_REGION')
    )
    
    # List available files in bucket to verify connection and permissions
    bucket = os.environ.get('S3_BUCKET')
    print(f'Listing files in bucket {bucket}:')
    response = s3_client.list_objects_v2(Bucket=bucket)
    if 'Contents' in response:
        for obj in response['Contents']:
            print(f'  {obj[\"Key\"]}')
    else:
        print('  No objects found in bucket')
    
    # Download the model
    model_name = os.environ.get('MODEL_TO_DOWNLOAD')
    model_path = '$MODEL_PATH'
    
    print(f'Downloading model {model_name} from bucket {bucket}')
    s3_client.download_file(bucket, model_name, model_path)
    
    if os.path.exists(model_path):
        print(f'Model downloaded to {model_path}, size: {os.path.getsize(model_path)} bytes')
    else:
        print(f'ERROR: Model file not found at {model_path} after download')
        sys.exit(1)
        
except ClientError as e:
    print(f'ERROR: S3 client error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'ERROR: Failed to download model: {e}')
    sys.exit(1)
"
  
  # Verify the model was downloaded
  if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE=$(ls -lh "$MODEL_PATH" | awk '{print $5}')
    echo "Model downloaded successfully: $MODEL_PATH (size: $MODEL_SIZE)"
  else
    echo "ERROR: Model download failed! File not found at $MODEL_PATH"
    echo "Contents of models directory:"
    ls -la /app/models/
    exit 1
  fi
else
  echo "ERROR: Missing required environment variables:"
  echo "  MODEL_TO_DOWNLOAD: $MODEL_TO_DOWNLOAD"
  echo "  S3_BUCKET: $S3_BUCKET"
  exit 1
fi

# Start the app
echo "Starting Monet Style Transfer Application..."
exec python app/app.py 