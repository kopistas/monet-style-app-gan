import os
import io
import base64
import sys
import logging
import time
import gc
import traceback
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import torch
from PIL import Image
import torchvision.transforms as T
import requests
from dotenv import load_dotenv
import random

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
ENV_MODEL_PATH = os.getenv('MODEL_PATH', '')
MODEL_PATH = ENV_MODEL_PATH if ENV_MODEL_PATH else "/app/models/photo2monet_cyclegan_mark4.pt"
UPLOAD_FOLDER = 'app/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UNSPLASH_API_KEY = os.getenv('UNSPLASH_API_KEY', '')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# Model variable initialized as None
gen = None
model_loading = False
model_load_error = None  # Track any model loading errors

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
    app_models_path = "/app/models/photo2monet_cyclegan_mark4.pt"
    alt_exists = os.path.exists(app_models_path)
    logger.info(f"Model at {app_models_path}: {'EXISTS' if alt_exists else 'NOT FOUND'}")
    
    root_model_path = "photo2monet_cyclegan_mark4.pt"
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
    """Load the model on demand with enhanced error handling"""
    global gen, model_loading, model_load_error
    
    if gen is not None:
        logger.info("Model already loaded")
        return True
    
    if model_loading:
        logger.info("Model is currently loading")
        return False
    
    model_loading = True
    model_load_error = None
    logger.info(f"Loading model from {MODEL_PATH}")
    
    try:
        # First check if model exists in any location
        model_exists, found_path = check_model_exists()
        
        if not model_exists:
            error_msg = f"Model file not found at {MODEL_PATH} or any alternative locations"
            logger.error(error_msg)
            model_load_error = error_msg
            model_loading = False
            return False
        
        # Log model file size for debugging
        try:
            model_size = os.path.getsize(found_path) / (1024 * 1024)  # Size in MB
            logger.info(f"Model file size: {model_size:.2f} MB")
        except Exception as e:
            logger.warning(f"Couldn't get model file size: {e}")
        
        # Load the model
        logger.info(f"Loading model from {found_path}")
        gen = torch.jit.load(found_path, map_location=DEVICE).eval()
        
        # Log model info
        try:
            logger.info(f"Model loaded successfully. Device: {next(gen.parameters()).device}")
            logger.info(f"Model memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB (allocated)")
        except Exception as e:
            logger.warning(f"Error getting model details: {e}")
            
        model_loading = False
        return True
        
    except Exception as e:
        tb = traceback.format_exc()
        error_msg = f"Error loading model: {e}\n{tb}"
        logger.error(error_msg)
        model_load_error = error_msg
        model_loading = False
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


# ---------- new process_image ----------------------------------------
def process_image(image: Image.Image):
    """
    • ≤256 px  → fast path (single crop).
    • >256 px → CPU tiling with 256-px tiles.
    Returns (result-dict, None) or (None, error-msg).
    """
    global gen, model_load_error

    if gen is None and not load_model():
        error_message = "Ошибка загрузки модели. "
        if model_load_error:
            error_message += f"Подробности: {model_load_error}"
        else:
            error_message += "Пожалуйста, повторите запрос через несколько секунд."
        return None, error_message

    try:
        t0 = time.time()

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

        # Clear RAM after processing
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "original":  base64.b64encode(buf_in.getvalue()).decode(),
            "generated": base64.b64encode(buf_out.getvalue()).decode(),
            "processing_time": f"{proc_time:.2f}"
        }, None

    except Exception as e:
        tb = traceback.format_exc()
        error_msg = f"Ошибка при обработке изображения: {e}\n{tb}"
        logger.exception("Error processing image")
        
        # Clear RAM after error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return None, error_msg
     
@app.route('/')
def index():
    logger.info("Index page requested")
    return render_template('index.html')

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """Check if the model is loaded or currently loading"""
    global gen, model_loading, model_load_error
    
    if model_loading:
        status = "loading"
    elif gen is not None:
        status = "loaded"
    else:
        status = "unloaded"
    
    # Check if model exists
    model_exists, found_path = check_model_exists()
    
    # Check for potential memory issues
    memory_info = {}
    if torch.cuda.is_available():
        memory_info["cuda_allocated"] = f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
        memory_info["cuda_reserved"] = f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB"
        memory_info["cuda_max_memory"] = f"{torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    
    free_mem = 0
    try:
        import psutil
        free_mem = psutil.virtual_memory().available / (1024**3)  # GB
    except ImportError:
        free_mem = "psutil not installed"
        
    return jsonify({
        "status": status,
        "device": DEVICE,
        "error": model_load_error,
        "model_path": MODEL_PATH,
        "model_exists": model_exists,
        "found_at": found_path if model_exists else None,
        "memory": memory_info,
        "system_free_memory_gb": free_mem
    })

@app.route('/api/transform', methods=['POST'])
def transform_image():
    logger.info("Transform endpoint called")
    try:
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
        logger.error(traceback.format_exc())
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
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Ошибка при загрузке изображения по URL: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Error transforming image from URL: {e}")
        logger.error(traceback.format_exc())
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
    global gen, model_loading, model_load_error
    
    status = "loading" if model_loading else ("loaded" if gen is not None else "unloaded")
    
    # Check if model exists
    model_exists, _ = check_model_exists()
    
    return jsonify({
        "status": "ok", 
        "model_status": status,
        "model_exists": model_exists,
        "model_path": MODEL_PATH,
        "model_error": model_load_error, 
        "message": "Сервер работает"
    })

if __name__ == '__main__':
    print("Запуск сервера трансформации изображений в стиле Моне...")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    print(f"Сервер доступен на порту 5080. Откройте http://localhost:5080 в браузере")
    logger.info(f"Сервер запускается на порту 5080")
    app.run(host='0.0.0.0', port=5080)