<p align="center">
  <img src="repo/logo.png" width="400" alt="Monet Style Transfer Logo">
</p>

<p align="center">
  <a href="https://github.com/kopistas/monet-style-app-gan/actions"><img src="https://github.com/kopistas/monet-style-app-gan/workflows/Build%20and%20Push%20Docker%20Image/badge.svg" alt="Build Status"></a>
  <a href="https://github.com/kopistas/monet-style-app-gan/issues"><img src="https://img.shields.io/github/issues/kopistas/monet-style-app-gan.svg" alt="Issues"></a>
  <a href="https://github.com/kopistas/monet-style-app-gan/blob/main/LICENSE"><img src="https://img.shields.io/github/license/kopistas/monet-style-app-gan.svg" alt="License"></a>
  <a href="https://github.com/kopistas/monet-style-app-gan/stargazers"><img src="https://img.shields.io/github/stars/kopistas/monet-style-app-gan.svg" alt="Stars"></a>
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.0+-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/kubernetes-1.25+-green.svg" alt="Kubernetes">
  <img src="https://img.shields.io/badge/terraform-1.0+-purple.svg" alt="Terraform">
</p>

# 🎨 Monet Style Transfer App

Это веб-приложение позволяет пользователям трансформировать свои фотографии в стиль картин Клода Моне с использованием модели CycleGAN. Данный репозиторий является простым примером как работы с моделями CycleGAN, так и настройки среды для развертывания приложения с моделью в продакшене.

## ✨ Features of Web App

- Загрузка собственных изображений или выбор из Unsplash
- Трансформация в стиль Моне в реальном времени
- Динамическая загрузка моделей с автоматическими обновлениями из S3
- Красивая визуализация прогресса

## 🏗️ Infrastructure

На момент начала работы над проектом в репозитории был реализован StyleGAN и веб приложение для демонстрации работы модели. 

В рамках работы над MLOps было реализовано ("по одному пушу"): 

- 🌍 Развертывание/обновление инфраструктуры в Digital Ocean с использованием Terraform:
  - Создается хранилище S3
  - Разворачивается Load Balancer
  - Разворачивается кластер Kubernetes
  - Создаются доменные имена для MLFlow и Web App Endpoint
  - Прописываются записи для DNS имен на Load Balancer, который уже дальше разруливает это на конкретные сервисы
- 🔄 В Kubernetes: 
  - Выписывается TLS сертификаты для созданных доменных имен через cert-manager
  - Разворачиваются сервисы MLflow и веб-приложение
  - Настроен NGINX Ingress Controller для маршрутизации запросов к сервисам
- 📊 MLflow: 
  - Закрыт Basic Auth через NGINX аннотации в Ingress
  - Подключено хранилище S3
  - Хранит логи тренировочных сессий
  - Выдает модели по алиасу
- 🖼️ Веб-приложение: 
  - Делает инференцию
  - Скачивает модель с MLflow по алиасу, поддерживает "hot swap" - стоит перевесить алиас в MLflow на другую модель, приложение скачает новую модель
  - Красиво выглядит
- 🚀 СI/CD: 
  - Все происходит и настраивается по пушу в main, для этого настроены Github Actions. 
  - При первоначальном запуске в MLflow загружается базовая модель. 

## 📂 Структура проекта

```
monet-style-app-gan/
├── app/                # Приложение Flask с шаблонами и статикой
├── training/           # Скрипты для обучения CycleGAN и логирования в MLflow
├── kubernetes/         # Конфигурации для K8s (web-app, mlflow, cert-manager)
├── terraform/          # Инфраструктурный код
├── scripts/            # Вспомогательные скрипты
├── repo/               # Ресурсы репозитория (логотип и т.д.)
└── .github/            # Конфигурация CI/CD в GitHub Actions
```

## 🛠️ Technical Stack

- Фронтенд: HTML, CSS, JavaScript
- Бэкенд: Flask (Python)
- Нейронная сеть: PyTorch
- MLOps: MLflow
- Контейнеризация: Docker
- Оркестрация: Kubernetes
- CI/CD: GitHub Actions
- IaC: Terraform
- Облако: DigitalOcean (Kubernetes, Spaces)

## Local Deployment 

Веб-приложение можно запустить локально в докере. Для этого нужно склонировать репу, в корень положить .env и сделать docker compose up -d.

## 📄 License

Лицензия MIT 