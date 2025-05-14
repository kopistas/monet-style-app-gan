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

# Copy the model and app code
COPY photo2monet_fastcut_mark2.pt .
COPY photo2monet_cyclegan_mark3.pt .

COPY app/ ./app/

# Verify files are copied correctly
RUN ls -la && ls -la app/

# Create uploads directory
RUN mkdir -p app/uploads

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 5080

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5080/health || exit 1

# Run the application
CMD ["python", "app/app.py"] 