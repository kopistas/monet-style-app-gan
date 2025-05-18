#!/bin/bash

# This script helps test the model promotion functionality locally
# Usage: ./scripts/test_model_promotion.sh <model_path>

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <model_path>"
  echo "Example: $0 photo2monet_cyclegan_mark4.pt"
  exit 1
fi

# Check if required environment variables are set
if [ -z "$S3_BUCKET" ] || [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
  echo "Please set the following environment variables:"
  echo "export S3_BUCKET=your-bucket-name"
  echo "export AWS_ACCESS_KEY_ID=your-access-key"
  echo "export AWS_SECRET_ACCESS_KEY=your-secret-key"
  echo "export S3_ENDPOINT=https://your-region.digitaloceanspaces.com  # Optional"
  exit 1
fi

MODEL_PATH=$1
MODEL_VERSION=$(date +%Y%m%d%H%M%S)

# Run the model promotion script
echo "Promoting model: $MODEL_PATH"
echo "Version: $MODEL_VERSION"
echo "S3 Bucket: $S3_BUCKET"

# Execute the actual promotion
python scripts/promote_model.py "$MODEL_PATH" \
  --bucket "$S3_BUCKET" \
  --version "$MODEL_VERSION" \
  --model-name "monet-style-transfer"

echo ""
echo "Model promotion complete!"
echo "The web app will automatically detect and download this new model on the next request."
echo ""
echo "To verify, visit the web app and check the model loading progress." 