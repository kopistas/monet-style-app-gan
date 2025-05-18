FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies with S3 support
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir boto3

# Copy the app code
COPY app/ ./app/

# Create uploads directory
RUN mkdir -p app/uploads

# Create models directory
RUN mkdir -p models

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/photo2monet_cyclegan_mark4.pt

# Expose the port
EXPOSE 5080

# Copy entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:5080/health || exit 1

# Run the application
ENTRYPOINT ["/app/entrypoint.sh"] 