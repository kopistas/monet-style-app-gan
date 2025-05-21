"""
Example script for training a GAN model in Google Colab and logging to MLflow

This script demonstrates how to:
1. Connect to an external MLflow server
2. Train a simple model (placeholder for the actual GAN training)
3. Log metrics, parameters, and artifacts
4. Register the model in MLflow

To use in Google Colab:
1. Upload this file to your Colab notebook
2. Install the required dependencies
3. Set the environment variables for MLflow and S3
4. Run the script
"""

import os
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import boto3
from botocore.client import Config
from torchvision import transforms
import time

# Parse arguments
parser = argparse.ArgumentParser(description='Train a model and log to MLflow')
parser.add_argument('--mlflow-tracking-uri', type=str, required=True,
                    help='MLflow tracking URI')
parser.add_argument('--experiment-name', type=str, default='monet-gan',
                    help='MLflow experiment name')
parser.add_argument('--model-name', type=str, default='MonetGAN',
                    help='Name for registered model')
parser.add_argument('--s3-endpoint-url', type=str, required=True,
                    help='S3 endpoint URL')
parser.add_argument('--s3-bucket', type=str, required=True,
                    help='S3 bucket name')
parser.add_argument('--aws-access-key-id', type=str, required=True,
                    help='AWS access key ID')
parser.add_argument('--aws-secret-access-key', type=str, required=True,
                    help='AWS secret access key')
parser.add_argument('--aws-region', type=str, required=True,
                    help='AWS region')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Batch size for training')
parser.add_argument('--learning-rate', type=float, default=0.0002,
                    help='Learning rate')

args = parser.parse_args()

# Configure MLflow
mlflow.set_tracking_uri(args.mlflow_tracking_uri)
print(f"MLflow tracking URI: {args.mlflow_tracking_uri}")

# Configure S3 client for artifact storage
os.environ['MLFLOW_S3_ENDPOINT_URL'] = args.s3_endpoint_url
os.environ['AWS_ACCESS_KEY_ID'] = args.aws_access_key_id
os.environ['AWS_SECRET_ACCESS_KEY'] = args.aws_secret_access_key
os.environ['AWS_REGION'] = args.aws_region

# Test S3 connection
s3_client = boto3.client(
    's3',
    endpoint_url=args.s3_endpoint_url,
    aws_access_key_id=args.aws_access_key_id,
    aws_secret_access_key=args.aws_secret_access_key,
    config=Config(signature_version='s3v4'),
    region_name=args.aws_region
)

try:
    response = s3_client.list_buckets()
    print("S3 connection successful")
    print(f"Available buckets: {[bucket['Name'] for bucket in response['Buckets']]}")
except Exception as e:
    print(f"Error connecting to S3: {e}")
    exit(1)

# Create or get MLflow experiment
try:
    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(
            args.experiment_name,
            artifact_location=f"s3://{args.s3_bucket}/mlflow-artifacts"
        )
    print(f"Using experiment '{args.experiment_name}' with ID: {experiment_id}")
except Exception as e:
    print(f"Error setting up MLflow experiment: {e}")
    exit(1)

# Define simple generator model for example
class SimpleGenerator(nn.Module):
    def __init__(self, input_dim=100, output_channels=3):
        super(SimpleGenerator, self).__init__()
        self.input_dim = input_dim
        
        self.model = nn.Sequential(
            # Initial projection and reshape
            nn.Linear(input_dim, 128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (128, 8, 8)),
            
            # Upsampling layers
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(16, output_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)

def train_model():
    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"Started MLflow run with ID: {run_id}")
        
        # Log parameters
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "model_type": "SimpleGenerator",
            "input_dim": 100,
            "output_channels": 3
        })
        
        # Initialize model
        generator = SimpleGenerator(input_dim=100, output_channels=3)
        optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
        criterion = nn.MSELoss()
        
        # Simulated training loop
        print("Starting training...")
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            for batch in range(10):  # Simulated batches
                # Generate random noise as input
                noise = torch.randn(args.batch_size, 100)
                
                # Generate images
                fake_images = generator(noise)
                
                # Simulated loss calculation
                target = torch.zeros_like(fake_images)
                loss = criterion(fake_images, target)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Simulate some metrics
                mlflow.log_metrics({
                    "batch_loss": loss.item(),
                    "generator_lr": optimizer.param_groups[0]['lr']
                }, step=epoch * 10 + batch)
            
            # Log epoch metrics
            avg_loss = epoch_loss / 10
            mlflow.log_metrics({
                "epoch": epoch,
                "avg_loss": avg_loss,
                "fake_image_mean": fake_images.mean().item(),
                "fake_image_std": fake_images.std().item()
            }, step=epoch)
            
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
            
            # Generate and log a sample image every few epochs
            if (epoch + 1) % 2 == 0 or epoch == args.epochs - 1:
                sample_noise = torch.randn(1, 100)
                with torch.no_grad():
                    sample_image = generator(sample_noise)
                    
                # Convert to PIL image and log
                img_tensor = (sample_image[0] + 1) / 2  # Normalize from [-1, 1] to [0, 1]
                img_tensor = img_tensor.detach().cpu()
                img_path = f"sample_epoch_{epoch+1}.png"
                transforms.ToPILImage()(img_tensor).save(img_path)
                mlflow.log_artifact(img_path)
                print(f"Logged sample image for epoch {epoch+1}")
        
        # Save the model
        model_path = "photo2monet_cyclegan.pt"
        torch.jit.script(generator).save(model_path)
        mlflow.log_artifact(model_path)
        print(f"Model saved and logged to MLflow")
        
        # Register the model
        mlflow.register_model(
            f"runs:/{run_id}/{model_path}",
            args.model_name
        )
        print(f"Model registered in MLflow as '{args.model_name}'")
        
        return run_id, model_path

if __name__ == "__main__":
    run_id, model_path = train_model()
    print(f"Training completed. Run ID: {run_id}")
    print(f"To promote this model to production, run:")
    print(f"python model-promotion.py --model-name {args.model_name} --model-version 1")

# Example usage in Colab:
"""
!pip install mlflow boto3 torch torchvision

!python mlflow_training_example.py \
    --mlflow-tracking-uri=https://mlflow.example.com \
    --experiment-name=monet-gan \
    --model-name=MonetGAN \
    --s3-endpoint-url=https://fra1.digitaloceanspaces.com \
    --s3-bucket=monet-models \
    --aws-access-key-id=YOUR_ACCESS_KEY \
    --aws-secret-access-key=YOUR_SECRET_KEY \
    --aws-region=fra1 \
    --epochs=10 \
    --batch-size=32 \
    --learning-rate=0.0002
""" 