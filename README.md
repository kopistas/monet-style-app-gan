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

# Трансформация в стиле Моне

Web-приложение для преобразования фотографий в стиль художника Клода Моне с использованием искусственного интеллекта.

## 📌 Особенности

- Загрузка изображений через drag-and-drop или выбор файла
- Выбор изображений из Unsplash
- Мгновенное преобразование фотографий в стиль Моне
- Адаптивный дизайн для ПК и мобильных устройств
- Оптимизированная загрузка модели по запросу для эффективного использования памяти
- MLflow для трекинга экспериментов и управления моделями
- Развертывание в Kubernetes с использованием Terraform и DigitalOcean
- Интеграция с S3 хранилищем (DO Spaces) для моделей
- Непрерывная интеграция и развертывание через GitHub Actions

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

## ⚡ Установка и использование

### Локальный запуск

1. Клонируйте репозиторий:
   ```
   git clone <url-репозитория>
   cd monet
   ```

2. Установите зависимости:
   ```
   pip install -r requirements.txt
   ```

3. Запустите приложение:
   ```
   python app/app.py
   ```

4. Откройте в браузере http://localhost:5080

### Запуск через Docker

1. Соберите и запустите Docker-контейнер:
   ```
   docker-compose up --build
   ```

2. Откройте в браузере http://localhost:5080

### Развертывание в Kubernetes

Полная инструкция по развертыванию в Kubernetes доступна в файле [README-infra.md](README-infra.md).

Основные шаги:

1. Настройте секреты GitHub для DigitalOcean и других сервисов
2. Запустите GitHub Actions workflow для создания инфраструктуры
3. Настройте DNS для вашего домена
4. Загрузите модель в S3 хранилище

## 🚀 MLflow и обучение моделей

Для обучения новых моделей и их регистрации в MLflow используйте скрипт `colab/mlflow_training_example.py`.

После обучения новой модели вы можете перевести ее в production с помощью скрипта `kubernetes/mlflow/model-promotion.py`.

## 🔄 Непрерывная интеграция (CI/CD)

Проект настроен для автоматической сборки и публикации Docker-образа, а также развертывания инфраструктуры при внесении изменений в репозиторий.

## 📄 Лицензия

MIT License 