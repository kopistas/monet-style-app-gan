# Monet Style Transfer Infrastructure

This repository contains the infrastructure code for deploying and managing a complete Monet style transfer application with MLflow integration in DigitalOcean Kubernetes Service (DOKS).

## Architecture Overview

The system consists of the following components:

1. **Web Application** - A Flask-based web service that provides style transfer capabilities using a PyTorch model
2. **MLflow Server** - For experiment tracking, model versioning, and model registry
3. **S3 Storage (DigitalOcean Spaces)** - For storing models and MLflow artifacts
4. **Kubernetes** - For container orchestration and scaling
5. **Terraform** - For infrastructure as code

![Architecture Diagram](repo/architecture.png)

## Prerequisites

- GitHub account (for GitHub Actions)
- DigitalOcean account
- A domain name (optional, but recommended for setting up HTTPS)

## Setup Instructions

### 1. Configure GitHub Repository Secrets

Add the following secrets to your GitHub repository:

- `DO_TOKEN` - DigitalOcean API token with write access
- `DO_SPACES_ACCESS_KEY` - DigitalOcean Spaces access key
- `DO_SPACES_SECRET_KEY` - DigitalOcean Spaces secret key
- `DO_REGION` - DigitalOcean region (e.g., `fra1`)
- `DO_SPACES_REGION` - DigitalOcean Spaces region (e.g., `fra1`)
- `DO_SPACES_NAME` - DigitalOcean Spaces bucket name (e.g., `monet-models`)
- `APP_DOMAIN` - Domain for the web app (e.g., `monet-app.example.com`)
- `MLFLOW_DOMAIN` - Domain for MLflow (e.g., `mlflow.example.com`)
- `EMAIL_FOR_SSL` - Email for Let's Encrypt SSL certificates
- `UNSPLASH_API_KEY` - Unsplash API key for the web app
- `USE_DO_DNS` - Set to `true` to automatically configure DNS using DigitalOcean DNS (optional)

### 2. Deploy Infrastructure

The infrastructure is automatically deployed when you push to the `main` branch or manually trigger the GitHub Actions workflow.

To manually trigger the workflow:
1. Go to your GitHub repository
2. Click on "Actions"
3. Select the "Deploy Infrastructure and Applications" workflow
4. Click "Run workflow"

### 3. DNS Configuration

#### Option 1: Automated DNS with DigitalOcean

If you're using DigitalOcean DNS for your domains and want automatic configuration:

1. Set the `USE_DO_DNS` GitHub secret to `true`
2. Ensure your domains are properly set up in DigitalOcean's DNS settings
3. The deployment will automatically configure A records for your domains

#### Option 2: Manual DNS Setup

If not using the automated DNS setup, you'll need to configure your DNS records manually:

1. Get the load balancer IP from the GitHub Actions output or by running:
   ```bash
   kubectl get svc -n ingress-nginx
   ```

2. Create A records for your domains in your DNS provider:
   - `monet-app.example.com` → Load balancer IP
   - `mlflow.example.com` → Load balancer IP

### 4. Using MLflow with Google Colab

A sample training script for Google Colab is provided in `colab/mlflow_training_example.py`. To use it:

1. Upload the script to your Colab notebook
2. Install the required dependencies:
   ```
   !pip install mlflow boto3 torch torchvision
   ```

3. Run the script with your MLflow and S3 configuration:
   ```
   !python mlflow_training_example.py \
       --mlflow-tracking-uri=https://mlflow.example.com \
       --experiment-name=monet-gan \
       --model-name=MonetGAN \
       --s3-endpoint-url=https://fra1.digitaloceanspaces.com \
       --s3-bucket=monet-models \
       --aws-access-key-id=YOUR_ACCESS_KEY \
       --aws-secret-access-key=YOUR_SECRET_KEY \
       --aws-region=fra1 \
       --epochs=10 \
       --batch-size=32 \
       --learning-rate=0.0002
   ```

### 5. Promoting Models to Production

To promote a model from MLflow to production and make the web app use it:

1. Create a Kubernetes job to run the promotion script:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: promote-model
  namespace: mlflow
spec:
  template:
    spec:
      containers:
      - name: promote-model
        image: python:3.9-slim
        command: ["bash", "-c"]
        args:
        - |
          pip install mlflow kubernetes boto3 && \
          curl -o model-promotion.py https://raw.githubusercontent.com/YOUR_USERNAME/monet/main/kubernetes/mlflow/model-promotion.py && \
          python model-promotion.py --model-name MonetGAN --model-version 1
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow.mlflow.svc.cluster.local"
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: spaces-credentials
              key: access_key
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: spaces-credentials
              key: secret_key
        - name: S3_ENDPOINT
          value: "https://fra1.digitaloceanspaces.com"
        - name: AWS_REGION
          value: "fra1"
        - name: MLFLOW_BUCKET
          value: "monet-models"
        - name: APP_BUCKET
          value: "monet-models"
      restartPolicy: Never
  backoffLimit: 2
```

2. Apply the job:
   ```bash
   kubectl apply -f promote-model-job.yaml
   ```

## Maintenance

### Updating the Web App

1. Make changes to the app code
2. Push to the `main` branch
3. GitHub Actions will automatically build and deploy the new version

### Scaling

To scale the web application, update the replica count in the Kubernetes deployment:

```bash
kubectl scale deployment monet-web-app -n monet-app --replicas=3
```

### Backup

The following components should be backed up:

1. MLflow database - Included in the persistent volume
2. DigitalOcean Spaces bucket - Contains models and artifacts
3. Terraform state - Stored in the Spaces bucket

## Troubleshooting

### Checking Pod Status

```bash
kubectl get pods -n monet-app
kubectl get pods -n mlflow
```

### Viewing Logs

```bash
kubectl logs -n monet-app deployment/monet-web-app
kubectl logs -n mlflow deployment/mlflow
```

### Accessing the MLflow UI

Open your MLflow domain in a browser (e.g., https://mlflow.example.com)

### Accessing the Web App

Open your app domain in a browser (e.g., https://monet-app.example.com) 