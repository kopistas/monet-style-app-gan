#!/usr/bin/env python3
"""
MLflow Model Promotion Script

This script promotes a registered model in MLflow to production and updates
the Kubernetes ConfigMap to make the web application use it.

Requirements:
- mlflow
- kubernetes
- boto3
"""

import os
import sys
import argparse
import logging
import mlflow
import boto3
from botocore.client import Config
from kubernetes import client, config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def promote_model(model_name, model_version):
    """Promote a model in MLflow to production."""
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow.mlflow.svc.cluster.local')
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    try:
        logger.info(f"Promoting model {model_name} version {model_version} to production")
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Production"
        )
        logger.info(f"Model {model_name} version {model_version} successfully promoted to production")
        
        # Get the artifact URI for the model
        model_version_details = client.get_model_version(
            name=model_name,
            version=model_version
        )
        artifact_uri = model_version_details.source
        
        # Return the artifact path within S3
        if artifact_uri.startswith('s3://'):
            # Remove s3:// prefix and bucket name
            bucket_and_path = artifact_uri.replace('s3://', '')
            bucket, path = bucket_and_path.split('/', 1)
            return path
        return None
    except Exception as e:
        logger.error(f"Error promoting model: {e}")
        return None

def copy_model_to_app_location(model_path, destination_path):
    """Copy the model from MLflow artifacts to the app's bucket location."""
    try:
        # Get S3 credentials from environment
        s3_endpoint = os.environ.get('S3_ENDPOINT')
        aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        region = os.environ.get('AWS_REGION')
        source_bucket = os.environ.get('MLFLOW_BUCKET')
        dest_bucket = os.environ.get('APP_BUCKET')
        
        # Configure S3 client
        s3 = boto3.client(
            's3',
            endpoint_url=s3_endpoint,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region,
            config=Config(signature_version='s3v4')
        )
        
        logger.info(f"Copying model from {source_bucket}/{model_path} to {dest_bucket}/{destination_path}")
        s3.copy_object(
            CopySource={'Bucket': source_bucket, 'Key': model_path},
            Bucket=dest_bucket,
            Key=destination_path
        )
        logger.info(f"Model successfully copied to {dest_bucket}/{destination_path}")
        return True
    except Exception as e:
        logger.error(f"Error copying model: {e}")
        return False

def update_kubernetes_configmap(model_path):
    """Update the Kubernetes ConfigMap with the new model path."""
    try:
        # Load Kubernetes configuration
        if os.path.exists('/var/run/secrets/kubernetes.io/serviceaccount/token'):
            # Inside the cluster
            config.load_incluster_config()
        else:
            # Outside the cluster
            config.load_kube_config()
        
        # Get the ConfigMap client
        v1 = client.CoreV1Api()
        
        # Update the ConfigMap
        namespace = 'monet-app'
        configmap_name = 'model-config'
        
        logger.info(f"Updating ConfigMap {configmap_name} in namespace {namespace} with model path {model_path}")
        
        # First, get the current ConfigMap
        configmap = v1.read_namespaced_config_map(configmap_name, namespace)
        
        # Update the model path
        configmap.data['promoted_model'] = model_path
        
        # Apply the update
        v1.replace_namespaced_config_map(configmap_name, namespace, configmap)
        
        logger.info(f"ConfigMap {configmap_name} successfully updated with model path {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error updating ConfigMap: {e}")
        return False

def main():
    """Main function to handle model promotion."""
    parser = argparse.ArgumentParser(description='Promote a model from MLflow to production')
    parser.add_argument('--model-name', required=True, help='Name of the model in MLflow')
    parser.add_argument('--model-version', required=True, help='Version of the model to promote')
    parser.add_argument('--dest-path', default='photo2monet_cyclegan_mark4.pt', 
                        help='Destination path in S3 (default: photo2monet_cyclegan_mark4.pt)')
    
    args = parser.parse_args()
    
    # Promote the model in MLflow
    artifact_path = promote_model(args.model_name, args.model_version)
    if not artifact_path:
        logger.error("Failed to promote model in MLflow")
        sys.exit(1)
    
    # Copy the model to the app's bucket
    if not copy_model_to_app_location(artifact_path, args.dest_path):
        logger.error("Failed to copy model to app location")
        sys.exit(1)
    
    # Update the Kubernetes ConfigMap
    if not update_kubernetes_configmap(args.dest_path):
        logger.error("Failed to update Kubernetes ConfigMap")
        sys.exit(1)
    
    logger.info(f"Model {args.model_name} version {args.model_version} successfully promoted and deployed to the app")

if __name__ == '__main__':
    main() 