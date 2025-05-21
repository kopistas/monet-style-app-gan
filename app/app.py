import os
import io
import base64
import sys
import logging
import time
import gc
import traceback
import json
import threading
import urllib.parse
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import torch
from PIL import Image
import torchvision.transforms as T
import requests
from dotenv import load_dotenv
import random
import boto3
from botocore.exceptions import ClientError

# Import MLflow for model management
try:
    import mlflow
    import mlflow.artifacts
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not installed. Model downloading from MLflow will not be available.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Log all environment variables related to model loading
logger.info(f"MODEL_PATH env: {os.getenv('MODEL_PATH', 'Not set')}")
logger.info(f"S3_BUCKET env: {os.getenv('S3_BUCKET', 'Not set')}")
logger.info(f"S3_ENDPOINT env: {os.getenv('S3_ENDPOINT', 'Not set')}")
logger.info(f"MODEL_TO_DOWNLOAD env: {os.getenv('MODEL_TO_DOWNLOAD', 'Not set')}")
logger.info(f"DEFAULT_MODEL_NAME env: {os.getenv('DEFAULT_MODEL_NAME', 'Not set')}")
logger.info(f"PROMOTED_MODEL_NAME env: {os.getenv('PROMOTED_MODEL_NAME', 'Not set')}")

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
logger.info("Flask app created with CORS support")

# Add custom error handler for all exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(traceback.format_exc())
    return jsonify({"error": f"Внутренняя ошибка сервера: {str(e)}"}), 500

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', 'photo2monet_cyclegan.pt')
UPLOAD_FOLDER = 'app/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UNSPLASH_API_KEY = os.getenv('UNSPLASH_API_KEY', '')
S3_BUCKET = os.getenv('S3_BUCKET', '')
MODEL_CHECK_INTERVAL = 3600  # Check for new models every hour

# Log configured model path and environment
logger.info(f"Configured MODEL_PATH: {MODEL_PATH}")
logger.info(f"S3_BUCKET: {S3_BUCKET}")
logger.info(f"S3_ENDPOINT: {os.getenv('S3_ENDPOINT', 'Not set')}")
logger.info(f"AWS_ACCESS_KEY_ID: {'Set' if os.getenv('AWS_ACCESS_KEY_ID') else 'Not set'}")
logger.info(f"AWS_SECRET_ACCESS_KEY: {'Set' if os.getenv('AWS_SECRET_ACCESS_KEY') else 'Not set'}")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# Model variables
gen = None
model_loading = False
model_load_error = None  # Track any model loading errors
model_download_progress = {"status": "idle", "message": ""}
inference_progress = {"status": "idle", "message": ""}
last_model_check_time = 0

# Set up transforms
prep = T.Compose([
    T.Resize(256, Image.BICUBIC),
    T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize((0.5,)*3, (0.5,)*3)
])
denorm = lambda x: x.mul(0.5).add(0.5).clamp(0, 1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_model_exists():
    """Check if the model file exists and log detailed diagnostics"""
    global MODEL_PATH
    
    # Check standard location
    standard_exists = os.path.exists(MODEL_PATH)
    logger.info(f"Model at {MODEL_PATH}: {'EXISTS' if standard_exists else 'NOT FOUND'}")
    
    # Check alternative locations
    app_models_path = "/app/models/photo2monet_cyclegan.pt"
    alt_exists = os.path.exists(app_models_path)
    logger.info(f"Model at {app_models_path}: {'EXISTS' if alt_exists else 'NOT FOUND'}")
    
    root_model_path = "photo2monet_cyclegan.pt"
    root_exists = os.path.exists(root_model_path)
    logger.info(f"Model at {root_model_path}: {'EXISTS' if root_exists else 'NOT FOUND'}")
    
    # Check directories
    current_dir = os.getcwd()
    logger.info(f"Current working directory: {current_dir}")
    
    try:
        # List contents of important directories
        logger.info(f"Contents of current directory: {os.listdir('.')}")
        if os.path.exists("/app/models"):
            logger.info(f"Contents of /app/models: {os.listdir('/app/models')}")
        if os.path.exists(os.path.dirname(MODEL_PATH)):
            logger.info(f"Contents of {os.path.dirname(MODEL_PATH)}: {os.listdir(os.path.dirname(MODEL_PATH))}")
    except Exception as e:
        logger.error(f"Error listing directories: {e}")
    
    # If model exists in any location, use that path
    if standard_exists:
        return True, MODEL_PATH
    elif alt_exists:
        MODEL_PATH = app_models_path
        return True, MODEL_PATH
    elif root_exists:
        MODEL_PATH = root_model_path
        return True, MODEL_PATH
    
    return False, None

def load_model():
    """Load the model on demand"""
    global gen, model_loading, inference_progress
    
    if gen is not None:
        logger.info("Model already loaded")
        return True
    
    if model_loading:
        logger.info("Model is currently loading")
        return False
    
    model_loading = True
    inference_progress = {"status": "loading", "message": "Loading model into memory"}
    logger.info(f"Loading model from {MODEL_PATH}")
    
    try:
        # Check for model in several locations
        model_locations = [
            MODEL_PATH,                                  # Default path from env var
            os.path.join(os.getcwd(), MODEL_PATH),       # Relative to current directory
            os.path.join(os.getcwd(), "models", os.path.basename(MODEL_PATH)),  # In models subdir
            os.path.join("/app", MODEL_PATH),            # Docker container root
            os.path.join("/app/models", os.path.basename(MODEL_PATH)),  # Docker container models dir
            os.path.join("/app", os.path.basename(MODEL_PATH))   # Dockerfile copied model to root
        ]
        
        # Find the first valid model file
        model_file = None
        for location in model_locations:
            if os.path.exists(location):
                model_file = location
                logger.info(f"Found model at {location}")
                break
                
        if not model_file:
            locations_str = "\n - ".join(model_locations)
            logger.error(f"Model file not found in any of these locations:\n - {locations_str}")
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Files in current directory: {os.listdir('.')}")
            if os.path.exists('/app'):
                logger.info(f"Files in /app: {os.listdir('/app')}")
            model_loading = False
            inference_progress = {"status": "error", "message": "Model file not found. Please check your configuration."}
            return False
        
        # Load the model
        gen = torch.jit.load(model_file, map_location=DEVICE).eval()
        logger.info(f"Model loaded successfully from {model_file}")
        model_loading = False
        inference_progress = {"status": "idle", "message": "Model loaded successfully"}
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_loading = False
        inference_progress = {"status": "error", "message": f"Error loading model: {e}"}
        return False

def unload_model():
    """Unload the model to free memory"""
    global gen
    
    if gen is None:
        return
    
    logger.info("Unloading model to free memory")
    gen = None
    # Force garbage collection to ensure memory is freed
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded")

to_tensor = T.ToTensor()
norm      = T.Normalize((0.5,)*3, (0.5,)*3)
eps       = 1e-6

# pre-compute a 2-D cosine window we'll reuse for every tile (keeps RAM tiny)
def _make_cosine_window(tile):
    ramp = torch.hann_window(tile, periodic=False)      # 1-D raised-cosine
    win  = torch.outer(ramp, ramp)                      # 2-D
    return win.unsqueeze(0)                             # shape (1, H, W)

COS_WIN_256 = _make_cosine_window(256)                 # cached weight map

def stylize_tiles_cpu(pil_img, tile=256, stride=224):
    """
    CPU tiler with 32-px overlap and cosine feathering → no seams.
    RAM impact is minimal (just one weight map the size of a tile).
    """
    global gen
    w, h   = pil_img.size
    canvas = torch.zeros(3, h, w)
    weight = torch.zeros(1, h, w)

    win = COS_WIN_256.to(canvas.dtype)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            crop = pil_img.crop((x, y, x+tile, y+tile))
            pad_w, pad_h = tile - crop.size[0], tile - crop.size[1]
            if pad_w or pad_h:                                       # edge pads
                crop = T.functional.pad(crop, (0, 0, pad_w, pad_h))

            tin = norm(to_tensor(crop)).unsqueeze(0)                 # CPU tensor
            with torch.no_grad():
                tout = gen(tin)[0]                                   # [-1,1]

            ow, oh = min(tile, w - x), min(tile, h - y)
            # multiply by window for feathering
            canvas[:, y:y+oh, x:x+ow] += tout[:, :oh, :ow] * win[:, :oh, :ow]
            weight[:, y:y+oh, x:x+ow] += win[:, :oh, :ow]

    blended = canvas / (weight + eps)
    return T.ToPILImage()(blended.clamp(-1, 1).add(1).mul(0.5))

def get_s3_client():
    """Create and return an S3 client"""
    s3_endpoint = os.getenv('S3_ENDPOINT', None)
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID', None)
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY', None)
    
    try:
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
            logger.warning("No S3 credentials provided, using anonymous client")
            return None
    except Exception as e:
        logger.error(f"Error creating S3 client: {e}")
        return None

def download_model_from_mlflow(local_path, alias="prod", fallback_alias="latest"):
    """Download model from MLflow with fallback to base alias and direct S3 if needed"""
    global model_download_progress
    
    if not MLFLOW_AVAILABLE:
        error_msg = "MLflow not installed, cannot download model"
        logger.error(error_msg)
        model_download_progress = {"status": "error", "message": error_msg}
        return False
    
    try:
        # Configuration
        registered_model_name = os.getenv("MLFLOW_MODEL_NAME", "style_transfer_photo2monet_cyclegan")
        artifact_name = os.getenv("MLFLOW_ARTIFACT_NAME", "photo2monet_cyclegan.pt")
        
        # MLflow configuration
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        mlflow_username = os.getenv("MLFLOW_USERNAME")
        mlflow_password = os.getenv("MLFLOW_PASSWORD")
        
        if not mlflow_tracking_uri:
            error_msg = "MLFLOW_TRACKING_URI environment variable not set"
            logger.error(error_msg)
            model_download_progress = {"status": "error", "message": error_msg}
            return False
        
        # Set initial progress
        model_download_progress = {
            "status": "preparing", 
            "message": f"Preparing to download model from MLflow (alias: {alias})"
        }
        
        # Auth if needed
        if mlflow_username and mlflow_password:
            os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
        
        logger.info(f"Connecting to MLflow at {mlflow_tracking_uri}")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        client = mlflow.tracking.MlflowClient()
        
        # Function to attempt download with specific alias
        def try_download_with_alias(current_alias):
            try:
                logger.info(f"Fetching model version with alias: {current_alias}")
                version = client.get_model_version_by_alias(registered_model_name, current_alias)
                run_id = version.run_id
                artifact_uri = f"runs:/{run_id}/models/{artifact_name}"
                
                logger.info(f"Trying MLflow artifact fetch from: {artifact_uri}")
                model_download_progress = {
                    "status": "downloading", 
                    "message": f"Downloading model from MLflow (run_id: {run_id})"
                }
                
                try:
                    downloaded_path = mlflow.artifacts.download_artifacts(artifact_uri)
                    logger.info(f"Downloaded via MLflow to: {downloaded_path}")
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    
                    # Copy to final location if needed
                    import shutil
                    shutil.copy(downloaded_path, local_path)
                    logger.info(f"Copied model to: {local_path}")
                    
                    model_download_progress = {
                        "status": "completed", 
                        "message": f"Download complete using {current_alias} alias"
                    }
                    return True
                
                except Exception as e:
                    logger.warning(f"MLflow download failed: {e}")
                    logger.info("Attempting direct S3 fallback...")
                    
                    # Step 1: Get full artifact URI for the run
                    run = client.get_run(run_id)
                    base_uri = run.info.artifact_uri
                    logger.info(f"Run artifact base URI: {base_uri}")
                    
                    # Step 2: Parse and build full path
                    parsed = urllib.parse.urlparse(base_uri)
                    bucket = parsed.netloc
                    prefix = parsed.path.lstrip("/")
                    full_key = f"{prefix}/models/{artifact_name}"
                    logger.info(f"S3 URI: s3://{bucket}/{full_key}")
                    
                    # Step 3: Download using boto3
                    s3_endpoint = os.getenv('MLFLOW_S3_ENDPOINT_URL', os.getenv('S3_ENDPOINT'))
                    
                    session = boto3.session.Session()
                    s3 = session.client(
                        service_name="s3",
                        endpoint_url=s3_endpoint,
                        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
                    )
                    
                    try:
                        logger.info(f"Downloading from S3: {bucket}/{full_key} to {local_path}")
                        model_download_progress = {
                            "status": "downloading", 
                            "message": f"Downloading model via S3 fallback"
                        }
                        
                        s3.download_file(bucket, full_key, local_path)
                        logger.info(f"Downloaded via S3 to: {local_path}")
                        
                        model_download_progress = {
                            "status": "completed", 
                            "message": f"Download complete via S3 fallback for {current_alias} alias"
                        }
                        return True
                    
                    except Exception as s3e:
                        logger.error(f"Could not download model from S3: {s3e}")
                        return False
            
            except Exception as alias_error:
                logger.warning(f"Could not find model with alias {current_alias}: {alias_error}")
                return False
            
        # Try with primary alias
        if try_download_with_alias(alias):
            return True
            
        # Try with fallback alias if primary failed
        logger.info(f"Trying fallback alias: {fallback_alias}")
        if try_download_with_alias(fallback_alias):
            return True
            
        # If we get here, both aliases failed
        error_msg = f"Could not download model with any alias (tried: {alias}, {fallback_alias})"
        logger.error(error_msg)
        model_download_progress = {"status": "error", "message": error_msg}
        return False
        
    except Exception as e:
        error_msg = f"Error in MLflow download process: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        model_download_progress = {"status": "error", "message": error_msg}
        return False

def get_production_model_info():
    """Get information about the current production model from S3"""
    try:
        if not S3_BUCKET:
            logger.warning("S3_BUCKET environment variable not set, skipping model check")
            return None

        s3_client = get_s3_client()
        if not s3_client:
            logger.warning("Could not create S3 client, skipping model check")
            return None
            
        # Look for a production marker file that contains the current production model name
        # This could come from MLflow or our direct promotion script
        production_marker_key = "production_model.json"
        
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=production_marker_key)
            production_info = json.loads(response['Body'].read().decode('utf-8'))
            
            logger.info(f"Found production model info: {production_info}")
            return production_info
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"No production marker file found in bucket {S3_BUCKET}")
            else:
                logger.error(f"Error accessing production marker: {e}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting production model info: {e}")
        return None

def download_model_from_s3(model_key, local_path):
    """Download model from S3 with progress tracking"""
    global model_download_progress
    
    try:
        if not S3_BUCKET:
            error_msg = "S3_BUCKET environment variable not set"
            logger.error(error_msg)
            model_download_progress = {"status": "error", "message": error_msg}
            return False

        s3_client = get_s3_client()
        if not s3_client:
            error_msg = "Could not create S3 client"
            logger.error(error_msg)
            model_download_progress = {"status": "error", "message": error_msg}
            return False
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Update progress status
        model_download_progress = {
            "status": "downloading", 
            "message": f"Downloading model {model_key}"
        }
        
        # Custom download
        with open(local_path, 'wb') as f:
            logger.info(f"Downloading model from s3://{S3_BUCKET}/{model_key} to {local_path}")
            
            try:
                s3_object = s3_client.get_object(Bucket=S3_BUCKET, Key=model_key)
                
                # Stream the file
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = s3_object['Body'].read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                
                model_download_progress = {
                    "status": "completed", 
                    "message": "Download complete"
                }
                logger.info(f"Model download complete: {local_path}")
                return True
                
            except ClientError as e:
                error_msg = f"Error downloading model: {e}"
                logger.error(error_msg)
                model_download_progress = {"status": "error", "message": error_msg}
                return False
                
    except Exception as e:
        error_msg = f"Error in download process: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        model_download_progress = {"status": "error", "message": error_msg}
        return False

def check_and_update_model():
    """Check if a new model is available from MLflow or S3 and download it if needed"""
    global MODEL_PATH, last_model_check_time, model_download_progress
    
    # Don't check too frequently
    current_time = time.time()
    if current_time - last_model_check_time < MODEL_CHECK_INTERVAL:
        return False
        
    last_model_check_time = current_time
    
    # Create local path for downloaded model
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # First try MLflow if configured
    if MLFLOW_AVAILABLE and os.getenv("MLFLOW_TRACKING_URI"):
        logger.info("Checking for model updates from MLflow")
        
        # Extract model filename from MLflow configuration
        artifact_name = os.getenv("MLFLOW_ARTIFACT_NAME", "photo2monet_cyclegan.pt")
        local_model_path = os.path.join(models_dir, artifact_name)
            
        # Try prod alias first, fall back to base
        if download_model_from_mlflow(local_model_path, alias="prod", fallback_alias="base"):
            logger.info(f"Downloaded new model from MLflow to {local_model_path}")
            # Update model path to use the new model
            MODEL_PATH = local_model_path
                
            # Unload the current model if it's loaded
            unload_model()
                
            return True
        else:
            logger.warning("Failed to download model from MLflow, will try S3 fallback")
    else:
        logger.info("MLflow not configured, skipping MLflow model check")
    
    # Fall back to S3 if MLflow failed or isn't configured
    if not S3_BUCKET:
        logger.warning("S3_BUCKET environment variable not set, skipping model check")
        return False
    
    # Get current production model info from S3
    production_info = get_production_model_info()
    
    if not production_info or 'model_key' not in production_info:
        logger.warning("No production model information available from S3")
        return False
        
    production_model_key = production_info['model_key']
    model_version = production_info.get('version', 'unknown')
    
    # Extract model filename from S3 key
    model_filename = os.path.basename(production_model_key)
    local_model_path = os.path.join(models_dir, model_filename)
    
    # Check if we already have this model version
    current_model_name = os.path.basename(MODEL_PATH)
    
    if current_model_name == model_filename and os.path.exists(MODEL_PATH):
        logger.info(f"Already using the latest production model from S3: {model_filename}")
        return False
        
    logger.info(f"New production model available from S3: {production_model_key} (version: {model_version})")
    logger.info(f"Current model: {MODEL_PATH}, new model will be: {local_model_path}")
    
    # Download the new model
    model_download_progress = {
        "status": "preparing", 
        "message": f"Preparing to download new model from S3 (version: {model_version})"
    }
    
    # Download the model (this will update the progress)
    if download_model_from_s3(production_model_key, local_model_path):
        logger.info(f"Downloaded new model from S3 to {local_model_path}")
        # Update model path to use the new model
        MODEL_PATH = local_model_path
        
        # Unload the current model if it's loaded
        unload_model()
        
        return True
    else:
        logger.error("Failed to download new model from S3")
        return False

def process_image(image: Image.Image):
    """
    • ≤256 px  → fast path (single crop).
    • >256 px → CPU tiling with 256-px tiles.
    Returns (result-dict, None) or (None, error-msg).
    """
    global gen, inference_progress
    
    # First, check if there's a new model to download (but don't block)
    if not model_loading:
        thread = threading.Thread(target=check_and_update_model)
        thread.daemon = True
        thread.start()
    
    # Set initial progress - simplified without progress percentage
    inference_progress = {"status": "loading", "message": "Loading model"}

    # Load model if needed
    if gen is None and not load_model():
        error_message = "Идет загрузка модели... Пожалуйста, повторите запрос через несколько секунд."
        inference_progress = {"status": "error", "message": error_message}
        return None, error_message

    try:
        t0 = time.time()
        inference_progress = {"status": "processing", "message": "Processing image"}

        # Check image dimensions to prevent memory issues
        w, h = image.size
        max_dim = 1920  # Maximum dimension to process
        
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            logger.info(f"Resizing large image from {w}x{h} to {new_w}x{new_h}")
            image = image.resize((new_w, new_h), Image.LANCZOS)
        
        if max(image.size) <= 256:                       # small: old path
            tin = prep(image).unsqueeze(0)               # stay on CPU
            with torch.no_grad():
                tout = gen(tin)[0]
            out_img = T.ToPILImage()(denorm(tout))
        else:                                            # large: tile path
            out_img = stylize_tiles_cpu(image, tile=256, stride=224)

        proc_time = time.time() - t0

        # ---- base-64 encode -----------------------------------------
        buf_in, buf_out = io.BytesIO(), io.BytesIO()
        image.save(buf_in, format="JPEG")
        out_img.save(buf_out, format="JPEG")
        
        # Unload model to free memory
        unload_model()

        inference_progress = {"status": "completed", "message": "Processing complete"}

        return {
            "original":  base64.b64encode(buf_in.getvalue()).decode(),
            "generated": base64.b64encode(buf_out.getvalue()).decode(),
            "processing_time": f"{proc_time:.2f}"
        }, None

    except Exception as e:
        logger.exception("Error processing image")
        error_msg = f"Ошибка при обработке изображения: {e}"
        inference_progress = {"status": "error", "message": error_msg}
        
        # Unload model to free memory after error
        unload_model()
        
        return None, error_msg
     
@app.route('/')
def index():
    logger.info("Index page requested")
    return render_template('index.html')

@app.route('/api/progress', methods=['GET'])
def get_progress():
    """Get the current progress of model download and inference"""
    global model_download_progress, inference_progress
    
    return jsonify({
        "model_download": model_download_progress,
        "inference": inference_progress
    })

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """Check if the model is loaded or currently loading"""
    global gen, model_loading, model_download_progress
    
    if model_loading:
        status = "loading"
    elif gen is not None:
        status = "loaded"
    else:
        status = "unloaded"
    
    # Check if model exists
    model_exists = os.path.exists(MODEL_PATH)
    
    return jsonify({
        "status": status,
        "device": DEVICE,
        "model_path": MODEL_PATH,
        "model_exists": model_exists,
        "download_progress": model_download_progress
    })

@app.route('/api/transform', methods=['POST'])
def transform_image():
    logger.info("Transform endpoint called")
    # Check if image was uploaded
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({'error': 'Не указан файл'}), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({'error': 'Файл не выбран'}), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        logger.warning(f"File type not allowed: {file.filename}")
        return jsonify({'error': 'Неподдерживаемый тип файла'}), 400
    
    try:
        # Open and process the image
        img = Image.open(file.stream).convert("RGB")
        logger.info(f"Processing image: {file.filename}")
        
        result, error = process_image(img)
        
        if error:
            return jsonify({'error': error}), 500
        
        logger.info("Image transformed successfully")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error transforming image: {e}")
        return jsonify({'error': f'Ошибка при обработке изображения: {str(e)}'}), 500

@app.route('/api/transform-url', methods=['POST'])
def transform_url():
    logger.info("Transform URL endpoint called")
    try:
        data = request.json
        
        if not data or 'url' not in data:
            logger.warning("No URL provided")
            return jsonify({'error': 'URL не указан'}), 400
        
        image_url = data['url']
        logger.info(f"Processing image from URL: {image_url}")
        
        # Download the image from the URL
        response = requests.get(image_url, stream=True, timeout=15)
        response.raise_for_status()
        
        # Open and process the image
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        result, error = process_image(img)
        
        if error:
            return jsonify({'error': error}), 500
        
        logger.info("Image from URL transformed successfully")
        return jsonify(result)
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image from URL: {e}")
        return jsonify({'error': f'Ошибка при загрузке изображения по URL: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Error transforming image from URL: {e}")
        return jsonify({'error': f'Ошибка при обработке изображения по URL: {str(e)}'}), 500

@app.route('/api/unsplash-photos', methods=['GET'])
def get_unsplash_photos():
    query = request.args.get('query', 'landscape')
    logger.info(f"Unsplash photos requested with query: {query}")
    
    try:
        if not UNSPLASH_API_KEY:
            logger.warning("No Unsplash API key provided, using default images")
            # If no API key, return some default image URLs
            default_urls = [
                "https://images.unsplash.com/photo-1506744038136-46273834b3fb",
                "https://images.unsplash.com/photo-1469474968028-56623f02e42e",
                "https://images.unsplash.com/photo-1501785888041-af3ef285b470",
                "https://images.unsplash.com/photo-1441974231531-c6227db76b6e",
                "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05",
                "https://images.unsplash.com/photo-1472214103451-9374bd1c798e"
            ]
            photos = [{"urls": {"regular": url}} for url in default_urls]
            return jsonify({"results": photos})
        
        # Make a request to the Unsplash API
        url = f"https://api.unsplash.com/search/photos"
        params = {
            "query": query,
            "per_page": 12,
            "client_id": UNSPLASH_API_KEY
        }
        
        logger.info(f"Requesting photos from Unsplash API")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"Received {len(data.get('results', []))} photos from Unsplash")
        return jsonify(data)
    
    except Exception as e:
        logger.error(f"Error fetching Unsplash photos: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Simple endpoint to check if server is running"""
    logger.info("Health check endpoint called")
    global gen, model_loading
    
    status = "loading" if model_loading else ("loaded" if gen is not None else "unloaded")
    model_exists = os.path.exists(MODEL_PATH)
    
    return jsonify({
        "status": "ok", 
        "model_status": status,
        "model_path": MODEL_PATH,
        "model_exists": model_exists,
        "message": "Сервер работает"
    })

# Print all environment variables for debugging
def log_environment_variables():
    """Log all relevant environment variables for debugging"""
    logger.info("=== Environment Variables ===")
    for var in [
        'MODEL_PATH', 
        'S3_BUCKET', 
        'S3_ENDPOINT', 
        'AWS_ACCESS_KEY_ID', 
        'AWS_SECRET_ACCESS_KEY', 
        'UNSPLASH_API_KEY',
        # MLflow variables
        'MLFLOW_TRACKING_URI',
        'MLFLOW_USERNAME',
        'MLFLOW_PASSWORD',
        'MLFLOW_MODEL_NAME',
        'MLFLOW_ARTIFACT_NAME',
        'MLFLOW_S3_ENDPOINT_URL'
    ]:
        logger.info(f"{var}: {'Set' if os.getenv(var) else 'Not set'}")
    logger.info("============================")

# Add this error handler for 404 errors to fix the bug in the logs
@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors gracefully"""
    logger.info(f"404 error: {request.path}")
    return jsonify({"error": "The requested URL was not found"}), 404

if __name__ == '__main__':
    print("Запуск сервера трансформации изображений в стиле Моне...")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs("models", exist_ok=True)  # Ensure models directory exists
    print(f"Сервер доступен на порту 5080. Откройте http://localhost:5080 в браузере")
    logger.info(f"Сервер запускается на порту 5080")
    log_environment_variables()
    
    # Check for model updates on startup
    if MLFLOW_AVAILABLE and os.getenv("MLFLOW_TRACKING_URI"):
        logger.info("MLflow is configured, checking for model updates on startup")
        threading.Thread(target=check_and_update_model, daemon=True).start()
    
    app.run(host='0.0.0.0', port=5080)