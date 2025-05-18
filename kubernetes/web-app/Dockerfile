FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts and app code
COPY scripts/ ./scripts/
COPY app/ ./app/

# Copy model file if it exists
COPY *.pt ./

# Create necessary directories
RUN mkdir -p app/uploads models

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/photo2monet_cyclegan_mark4.pt

# Make scripts executable
RUN chmod +x scripts/*.py

# Expose the port
EXPOSE 5080

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5080/health || exit 1

# Run the application
CMD ["python", "app/app.py"] 