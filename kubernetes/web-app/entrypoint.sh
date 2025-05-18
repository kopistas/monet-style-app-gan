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
fi

# Check for promoted model name from MLflow
if [ -n "$PROMOTED_MODEL_NAME" ]; then
  echo "Downloading promoted model $PROMOTED_MODEL_NAME from S3..."
  python -c "
import boto3
import os
from botocore.client import Config

# Configure S3 client
s3_client = boto3.client(
    's3',
    endpoint_url=os.environ.get('S3_ENDPOINT'),
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    config=Config(signature_version='s3v4'),
    region_name=os.environ.get('AWS_REGION')
)

# Download the model
bucket = os.environ.get('S3_BUCKET')
model_name = os.environ.get('PROMOTED_MODEL_NAME')
model_path = '/app/models/photo2monet_cyclegan_mark4.pt'

print(f'Downloading model {model_name} from bucket {bucket}')
s3_client.download_file(bucket, model_name, model_path)
print(f'Model downloaded to {model_path}')
"
else
  # Use the default model that's built into the image if available
  echo "No promoted model specified, using default model or downloading from S3..."
  
  if [ -n "$DEFAULT_MODEL_NAME" ] && [ -n "$S3_BUCKET" ]; then
    echo "Downloading default model $DEFAULT_MODEL_NAME from S3..."
    python -c "
import boto3
import os
from botocore.client import Config

# Configure S3 client
s3_client = boto3.client(
    's3',
    endpoint_url=os.environ.get('S3_ENDPOINT'),
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    config=Config(signature_version='s3v4'),
    region_name=os.environ.get('AWS_REGION')
)

# Download the model
bucket = os.environ.get('S3_BUCKET')
model_name = os.environ.get('DEFAULT_MODEL_NAME', 'photo2monet_cyclegan_mark4.pt')
model_path = '/app/models/photo2monet_cyclegan_mark4.pt'

print(f'Downloading model {model_name} from bucket {bucket}')
s3_client.download_file(bucket, model_name, model_path)
print(f'Model downloaded to {model_path}')
"
  fi
fi

# Start the app
echo "Starting Monet Style Transfer Application..."
exec python app/app.py 