<p align="center">
  <img src="repo/logo.png" width="400" alt="Monet Style Transfer Logo">
</p>

<p align="center">
  <a href="https://github.com/kopistas/monet-style-app-gan/actions"><img src="https://github.com/kopistas/monet-style-app-gan/workflows/Build%20and%20Push%20Docker%20Image/badge.svg" alt="Build Status"></a>
  <a href="https://github.com/kopistas/monet-style-app-gan/issues"><img src="https://img.shields.io/github/issues/kopistas/monet-style-app-gan.svg" alt="Issues"></a>
  <a href="https://github.com/kopistas/monet-style-app-gan/blob/main/LICENSE"><img src="https://img.shields.io/github/license/kopistas/monet-style-app-gan.svg" alt="License"></a>
  <a href="https://github.com/kopistas/monet-style-app-gan/stargazers"><img src="https://img.shields.io/github/stars/kopistas/monet-style-app-gan.svg" alt="Stars"></a>
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python">
</p>

# Monet Style Transfer App

This web application allows users to transform their photos into the style of Claude Monet paintings using a CycleGAN model.

## Features

- Upload your own images or choose from Unsplash
- Real-time transformation to Monet's style
- Dynamic model loading with automatic updates from S3
- Beautiful progress visualization

## Setup

### Environment Variables

The application uses the following environment variables:

```
# S3 Configuration
S3_BUCKET=your-model-bucket
S3_ENDPOINT=https://your-s3-endpoint.com  # Optional for custom endpoints
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Optional Settings
UNSPLASH_API_KEY=your-unsplash-api-key
MODEL_PATH=/app/models/custom-model.pt  # Optional local model path
```

### Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app/app.py
```

The application will be available at http://localhost:5080

## Model Management

The application has a dynamic model loading system with the following features:

1. **No pre-loaded models** - Models are loaded on-demand when needed
2. **Automatic updates** - The app checks for new production models in S3
3. **Hot-swapping** - New models are downloaded and applied without restarting
4. **Progress tracking** - Download and inference progress is shown in the UI

### How It Works

1. When a user requests an image transformation, the app checks if there's a newer production model available in S3
2. If a new model is found, it's downloaded to the local cache
3. The new model is loaded and used for inference
4. Progress is shown in the UI during all these steps

### Promoting a New Model to Production

To mark a model as "production ready" for the app to use:

```bash
# Upload and promote a model
python scripts/promote_model.py path/to/your/model.pt --bucket your-model-bucket

# With additional options
python scripts/promote_model.py path/to/your/model.pt \
    --bucket your-model-bucket \
    --model-key models/custom-name.pt \
    --version v1.2.3 \
    --run-id mlflow-run-id \
    --model-name "monet-style-transfer"
```

This script will:
1. Upload your model to S3
2. Create/update a `production_model.json` marker file
3. The web app will detect and download this new model on the next check

## Integration with MLflow

The model promotion script can fetch metadata from MLflow to include with the model:

1. Metrics like training loss, validation scores, etc.
2. Model version information
3. Tags and annotations

## Development

### Required Packages

```
flask
flask-cors
torch
torchvision
pillow
requests
python-dotenv
boto3
```

## 🛠️ Технический стек

- Фронтенд: HTML, CSS, JavaScript
- Бэкенд: Flask (Python)
- Нейронная сеть: PyTorch
- MLOps: MLflow
- Контейнеризация: Docker
- Оркестрация: Kubernetes
- CI/CD: GitHub Actions
- IaC: Terraform
- Облако: DigitalOcean (Kubernetes, Spaces)

## 🎯 Архитектура

![Архитектура](repo/architecture.png)

Система состоит из следующих компонентов:

1. **Kubernetes кластер** в DigitalOcean для запуска всех сервисов
2. **Веб-приложение** на Flask для преобразования фотографий
3. **MLflow сервер** для управления моделями и экспериментами
4. **S3 хранилище** (DO Spaces) для хранения моделей

## 🚀 MLflow и обучение моделей

Для обучения новых моделей и их регистрации в MLflow используйте скрипт `colab/mlflow_training_example.py`.

После обучения новой модели вы можете перевести ее в production с помощью скрипта `kubernetes/mlflow/model-promotion.py`.

## 🔄 Непрерывная интеграция (CI/CD)

Проект настроен для автоматической сборки и публикации Docker-образа, а также развертывания инфраструктуры при внесении изменений в репозиторий.

## 📄 Лицензия

MIT License 