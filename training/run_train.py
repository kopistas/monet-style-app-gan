#!/usr/bin/env python

import os
import sys
import glob
import random
import shutil
import subprocess
import torch
import zipfile
from pathlib import Path
from dotenv import load_dotenv
import mlflow

# ---- Load environment variables ----
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path, override=True)

# ---- CONFIGURATION ----
# Dataset settings
DATASET = "monet2photo"
DATASET_ROOT = "training/datasets"
MODELS_ROOT = "training/models"

# Training settings
EXP_NAME = "photo2monet_cyclegan"
CROP_SIZE = 256
BATCH_SIZE = 1
EPOCHS = 1
EPOCHS_DECAY = 0

USE_MLFLOW = True
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI')
MLFLOW_EXPERIMENT_NAME = "monet_cyclegan"
MLFLOW_USERNAME = os.environ.get('MLFLOW_USERNAME')  
MLFLOW_PASSWORD = os.environ.get('MLFLOW_PASSWORD')  
MLFLOW_ENABLE_AUTOLOG = True

# Device settings
USE_MPS = True  # Set to False to disable MPS even if available

# ---- SETUP PATHS ----
# Add CycleGAN submodule to path
CYCLEGAN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cyclegan")
sys.path.append(CYCLEGAN_PATH)

# Create directories if they don't exist
os.makedirs(DATASET_ROOT, exist_ok=True)
os.makedirs(MODELS_ROOT, exist_ok=True)
os.makedirs(os.path.join(DATASET_ROOT, DATASET), exist_ok=True)

# ---- DEVICE DETECTION ----
def get_device():
    """Determine the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        print("CUDA GPU detected!")
        return "cuda"
    elif USE_MPS and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Apple Silicon MPS detected!")
        return "mps"
    else:
        print("No GPU detected, using CPU.")
        return "cpu"

# ---- DATASET PREPARATION ----
def prepare_dataset():
    """Download and prepare the Monet2Photo dataset."""
    dataset_path = os.path.join(DATASET_ROOT, DATASET)
    
    # Check if dataset already exists
    if os.path.exists(os.path.join(dataset_path, "trainA")) and \
       os.path.exists(os.path.join(dataset_path, "trainB")):
        print(f"Dataset already exists at {dataset_path}")
        return
    
    print("Downloading Monet2Photo dataset...")
    # Use Kaggle API directly instead of subprocess
    try:
        import kaggle
        # Download the dataset with unzip=True
        kaggle.api.dataset_download_files(
            dataset='balraj98/monet2photo',
            path=dataset_path,
            unzip=True
        )
        print("Dataset downloaded and extracted successfully")
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please make sure the kaggle package is installed and configured with your API key")
        print("Make sure you have ~/.kaggle/kaggle.json with your API credentials")
        print("If not, create it with proper permissions: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)
    
    # Optional: Reduce trainB set to 1000 images as in the original script
    original_trainB = os.path.join(dataset_path, "trainB")
    backup_trainB = os.path.join(dataset_path, "trainB_full")
    
    if os.path.exists(original_trainB) and not os.path.exists(backup_trainB):
        # Backup original trainB
        shutil.move(original_trainB, backup_trainB)
        print(f"Original trainB moved to trainB_full")
        
        # Create fresh trainB folder
        os.makedirs(original_trainB, exist_ok=True)
        
        # Sample 1000 random images from trainB_full
        all_images = os.listdir(backup_trainB)
        sampled_images = random.sample(all_images, min(1000, len(all_images)))
        
        # Copy sampled images back to trainB
        for img in sampled_images:
            shutil.copy(os.path.join(backup_trainB, img), os.path.join(original_trainB, img))
            
        print(f"Reduced trainB ready with {len(sampled_images)} images")

# ---- TRAIN CYCLEGAN ----
def train_cyclegan():
    """Train CycleGAN model with MLFlow integration."""
    print("Starting CycleGAN training...")
    
    # Determine device
    device = get_device()
    gpu_ids = ""
    
    if device == "cuda":
        gpu_ids = "0"
    elif device == "mps":
        # MPS doesn't use the same gpu_ids format, but we pass a special flag
        gpu_ids = "mps"
    else:
        gpu_ids = "-1"  # CPU
    
    # Construct the command to run train.py with appropriate arguments
    train_script = os.path.join(CYCLEGAN_PATH, "train.py")
    
    cmd = [
        sys.executable,
        train_script,
        "--dataroot", os.path.join(DATASET_ROOT, DATASET),
        "--name", EXP_NAME,
        "--model", "cycle_gan",
        "--direction", "BtoA",  # Photo to Monet direction
        "--n_epochs", str(EPOCHS),
        "--n_epochs_decay", str(EPOCHS_DECAY),
        "--load_size", str(CROP_SIZE),
        "--crop_size", str(CROP_SIZE),
        "--batch_size", str(BATCH_SIZE),
        "--gpu_ids", gpu_ids,
        "--checkpoints_dir", MODELS_ROOT,
        "--save_epoch_freq", "1",  # Save every epoch
        "--save_latest_freq", "500",  # Save more frequently
        "--display_freq", "100",  # Display more frequently 
        "--print_freq", "100",  # Print more frequently
        "--display_id", "-1"  # Disable visdom display
    ]
    
    # Add MLFlow parameters if enabled
    if USE_MLFLOW:
        cmd.extend([
            "--use_mlflow",
            "--mlflow_tracking_uri", MLFLOW_TRACKING_URI,
            "--mlflow_experiment_name", MLFLOW_EXPERIMENT_NAME
        ])
        
        if MLFLOW_USERNAME and MLFLOW_PASSWORD:
            cmd.extend([
                "--mlflow_username", MLFLOW_USERNAME,
                "--mlflow_password", MLFLOW_PASSWORD
            ])
            
        if MLFLOW_ENABLE_AUTOLOG:
            cmd.extend([
                "--mlflow_autolog"
            ])
    
    # Set up MLflow environment variables
    env = os.environ.copy()
    if USE_MLFLOW and MLFLOW_USERNAME and MLFLOW_PASSWORD:
        env['MLFLOW_TRACKING_USERNAME'] = MLFLOW_USERNAME
        env['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_PASSWORD
    
    # Run the training command
    try:
        subprocess.run(cmd, check=True, env=env)
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        sys.exit(1)

# ---- EXPORT MODEL ----
def export_model():
    """Export the trained model to TorchScript format and log to MLflow."""
    print("Exporting model to TorchScript...")
    try:
        from cyclegan.models.networks import define_G
        import re
        from mlflow.tracking import MlflowClient

        # Paths
        ckpt_dir = os.path.join(MODELS_ROOT, EXP_NAME)
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*_net_G_A.pth")))

        if not ckpts:
            print("No checkpoints found. Make sure training completed successfully.")
            return

        ckpt_path = ckpts[-1]
        print(f"Exporting checkpoint: {ckpt_path}")

        # Initialize and load model
        netG = define_G(
            input_nc=3, output_nc=3, ngf=64,
            netG="resnet_9blocks", norm="instance",
            use_dropout=False, init_type="normal",
            init_gain=0.02, gpu_ids=[]
        ).cpu()

        state = torch.load(ckpt_path, map_location="cpu")
        state = state.get("state_dict", state)
        state = {re.sub(r"^(module\.|netG\.|G_A\.)", "", k): v for k, v in state.items()}

        missing, unexpected = netG.load_state_dict(state, strict=False)
        print(f"Loaded with {len(missing)} missing & {len(unexpected)} unexpected keys")

        # Trace TorchScript model
        netG.eval()
        dummy = torch.randn(1, 3, CROP_SIZE, CROP_SIZE)
        ts = torch.jit.trace(netG, dummy)

        # Save to models/ folder
        export_dir = "models"
        os.makedirs(export_dir, exist_ok=True)
        out_pt = os.path.join(export_dir, "photo2monet_cyclegan.pt")
        ts.save(out_pt)
        print(f"TorchScript saved to {out_pt}")

        if USE_MLFLOW:
            if MLFLOW_USERNAME and MLFLOW_PASSWORD:
                os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_USERNAME
                os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_PASSWORD

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

            active_run = mlflow.active_run()
            if active_run is None:
                run = mlflow.start_run(run_name=EXP_NAME)
            else:
                run = active_run

            try:
                # Log full model directory
                mlflow.log_artifacts(export_dir, artifact_path="models")

                # Add metadata
                mlflow.set_tag("model_type", "torchscript")
                mlflow.set_tag("input_size", f"{CROP_SIZE}x{CROP_SIZE}")
                mlflow.set_tag("checkpoint_source", os.path.basename(ckpt_path))

                # Register model from artifact dir
                model_uri = f"runs:/{run.info.run_id}/models"
                registered_model_name = f"style_transfer_{EXP_NAME}"
                print(f"Registering model from: {model_uri}")

                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name=registered_model_name
                )

                print(f"Model registered as '{registered_model_name}', version {registered_model.version}")

                # Update description + set alias
                client = MlflowClient()
                client.update_registered_model(
                    name=registered_model_name,
                    description=f"Monet style transfer model (CycleGAN) trained with {EPOCHS} epochs."
                )
                # client.set_registered_model_alias(
                #     name=registered_model_name,
                #     alias="prod",  # or "prod", "latest", etc.
                #     version=registered_model.version
                # )
                print(f"Alias 'prod' set for model version {registered_model.version}")

            finally:
                if active_run is None:
                    mlflow.end_run()

    except Exception as e:
        print(f"Error exporting model: {e}")

    """Export the trained model to TorchScript format and log to MLflow."""
    print("Exporting model to TorchScript...")
    
    try:
        # Import necessary modules from the CycleGAN repo
        from cyclegan.models.networks import define_G
        
        # Find the latest checkpoint
        ckpt_dir = os.path.join(MODELS_ROOT, EXP_NAME)
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*_net_G_A.pth")))
        
        if not ckpts:
            print("No checkpoints found. Make sure training completed successfully.")
            return
            
        ckpt_path = ckpts[-1]
        print(f"Exporting checkpoint: {ckpt_path}")
        
        # Initialize the generator network
        netG = define_G(
            input_nc=3, output_nc=3, ngf=64,
            netG="resnet_9blocks", norm="instance",
            use_dropout=False, init_type="normal",
            init_gain=0.02, gpu_ids=[]
        ).cpu()
        
        # Load the trained weights
        state = torch.load(ckpt_path, map_location="cpu")
        state = state.get("state_dict", state)
        
        # Clean up the state dict keys if needed
        import re
        state = {re.sub(r"^(module\.|netG\.|G_A\.)", "", k): v for k, v in state.items()}
        
        # Load the state dict
        missing, unexpected = netG.load_state_dict(state, strict=False)
        print(f"Loaded with {len(missing)} missing & {len(unexpected)} unexpected keys")
        
        # Trace and save as TorchScript
        netG.eval()
        dummy = torch.randn(1, 3, CROP_SIZE, CROP_SIZE)
        ts = torch.jit.trace(netG, dummy)
        
        # Save the traced model
        out_pt = os.path.join(MODELS_ROOT, f"{EXP_NAME}.pt")
        ts.save(out_pt)
        print(f"TorchScript saved to {out_pt}")
        
        # Log the model to MLflow if enabled
        if USE_MLFLOW:
            # Setup MLflow
            if MLFLOW_USERNAME and MLFLOW_PASSWORD:
                os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_USERNAME
                os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_PASSWORD
            
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            
            # Find the active run or create a new one
            active_run = mlflow.active_run()
            if active_run is None:
                run = mlflow.start_run(run_name=EXP_NAME)
            else:
                run = active_run
                
            try:
                # Log the model
                print(f"Logging TorchScript model to MLflow as artifact: {out_pt}")
                mlflow.log_artifact(out_pt, artifact_path="models")
                
                # Log additional metadata about the model
                mlflow.set_tag("model_type", "torchscript")
                mlflow.set_tag("input_size", f"{CROP_SIZE}x{CROP_SIZE}")
                mlflow.set_tag("checkpoint_source", os.path.basename(ckpt_path))
                
                # Register the model in MLflow Model Registry
                print("Registering model in MLflow Model Registry...")
                model_uri = f"runs:/{run.info.run_id}/models/{EXP_NAME}.pt"
                registered_model_name = f"style_transfer_{EXP_NAME}"
                
                try:
                    registered_model = mlflow.register_model(
                        model_uri=model_uri,
                        name=registered_model_name
                    )
                    print(f"Model registered with name: {registered_model_name}, version: {registered_model.version}")
                    
                    # Add model description
                    client = mlflow.tracking.MlflowClient()
                    client.update_registered_model(
                        name=registered_model_name,
                        description=f"Monet style transfer model (CycleGAN) trained with {EPOCHS} epochs."
                    )
                    
                    # Optional: transition to Production if this is a production model
                    # client.transition_model_version_stage(
                    #     name=registered_model_name,
                    #     version=registered_model.version,
                    #     stage="Production"
                    # )
                except Exception as reg_error:
                    print(f"Error registering model: {reg_error}")
                
                print(f"Model successfully logged to MLflow run: {run.info.run_id}")
            finally:
                # End the run if we started one
                if active_run is None:
                    mlflow.end_run()
        
    except Exception as e:
        print(f"Error exporting model: {e}")
        return

if __name__ == "__main__":
    print("CycleGAN Training Pipeline")
    prepare_dataset()
    train_cyclegan()
    export_model()
    print("Pipeline completed!")


