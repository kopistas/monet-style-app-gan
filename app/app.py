import os
import io
import base64
import sys
import logging
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

app = Flask(__name__, static_folder='static')
CORS(app)
logger.info("Flask app created with CORS support")

# Configuration
MODEL_PATH = "photo2monet_fastcut_mark2.pt"
UPLOAD_FOLDER = 'app/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UNSPLASH_API_KEY = os.getenv('UNSPLASH_API_KEY', '')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# Load the model
try:
    logger.info(f"Attempting to load model from {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Files in current directory: {os.listdir('.')}")
        gen = None
    else:
        gen = torch.jit.load(MODEL_PATH, map_location=DEVICE).eval()
        logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    gen = None

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

def process_image(image):
    """Process the image through the model and return base64 encoded results"""
    if gen is None:
        logger.error("Model not loaded, cannot process image")
        return None
    
    # Prepare the image
    tin = prep(image).unsqueeze(0).to(DEVICE)
    
    # Generate the Monet style image
    try:
        with torch.no_grad():
            tout = gen(tin)[0].cpu()
        
        # Convert to PIL Image
        out = T.ToPILImage()(denorm(tout))
        
        # Convert both original and generated images to base64
        buffered_original = io.BytesIO()
        image.save(buffered_original, format="JPEG")
        original_base64 = base64.b64encode(buffered_original.getvalue()).decode('utf-8')
        
        buffered_generated = io.BytesIO()
        out.save(buffered_generated, format="JPEG")
        generated_base64 = base64.b64encode(buffered_generated.getvalue()).decode('utf-8')
        
        logger.info("Image processed successfully")
        return {
            'original': original_base64,
            'generated': generated_base64
        }
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

@app.route('/')
def index():
    logger.info("Index page requested")
    return render_template('index.html')

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
        result = process_image(img)
        
        if result is None:
            logger.error("Model could not process the image")
            return jsonify({'error': 'Модель не может обработать изображение'}), 500
        
        logger.info("Image transformed successfully")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error transforming image: {e}")
        return jsonify({'error': f'Ошибка при обработке изображения: {str(e)}'}), 500

@app.route('/api/transform-url', methods=['POST'])
def transform_url():
    logger.info("Transform URL endpoint called")
    data = request.json
    
    if not data or 'url' not in data:
        logger.warning("No URL provided")
        return jsonify({'error': 'URL не указан'}), 400
    
    image_url = data['url']
    logger.info(f"Processing image from URL: {image_url}")
    
    try:
        # Download the image from the URL
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        
        # Open and process the image
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        result = process_image(img)
        
        if result is None:
            logger.error("Model could not process the image from URL")
            return jsonify({'error': 'Модель не может обработать изображение по URL'}), 500
        
        logger.info("Image from URL transformed successfully")
        return jsonify(result)
    
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
    return jsonify({"status": "ok", "model_loaded": gen is not None, "message": "Сервер работает"})

if __name__ == '__main__':
    print("Запуск сервера трансформации изображений в стиле Моне...")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    print(f"Сервер доступен на порту 5080. Откройте http://localhost:5080 в браузере")
    logger.info(f"Сервер запускается на порту 5080")
    app.run(host='0.0.0.0', port=5080)