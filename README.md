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

## Особенности

- Загрузка изображений через drag-and-drop или выбор файла
- Выбор изображений из Unsplash
- Мгновенное преобразование фотографий в стиль Моне
- Адаптивный дизайн для ПК и мобильных устройств
- Оптимизированная загрузка модели по запросу для эффективного использования памяти

## Технический стек

- Фронтенд: HTML, CSS, JavaScript
- Бэкенд: Flask (Python)
- Нейронная сеть: PyTorch
- Контейнеризация: Docker

## Установка и использование

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

## Непрерывная интеграция (CI/CD)

Проект настроен для автоматической сборки и публикации Docker-образа при внесении изменений в репозиторий.

### Публикация в GitHub Container Registry

По умолчанию настроена публикация в GitHub Container Registry (ghcr.io).

## Лицензия

MIT License 