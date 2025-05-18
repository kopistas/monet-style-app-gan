# Kubernetes Module for resources dependent on the cluster

# Instead of defining providers directly, we expect them to be passed from the root module
# The parent module should configure these providers with the appropriate connection details

# Create namespaces
resource "kubernetes_namespace" "mlflow" {
  count = var.deploy_mlflow ? 1 : 0
  metadata {
    name = "mlflow"
  }
}

resource "kubernetes_namespace" "monet_app" {
  metadata {
    name = "monet-app"
  }
}

# Create secrets for S3 access
resource "kubernetes_secret" "spaces_credentials_mlflow" {
  count = var.deploy_mlflow ? 1 : 0
  metadata {
    name      = "spaces-credentials"
    namespace = "mlflow"
  }

  data = {
    access_key = var.do_spaces_access_key
    secret_key = var.do_spaces_secret_key
  }

  depends_on = [kubernetes_namespace.mlflow]
}

resource "kubernetes_secret" "spaces_credentials_app" {
  metadata {
    name      = "spaces-credentials"
    namespace = "monet-app"
  }

  data = {
    access_key = var.do_spaces_access_key
    secret_key = var.do_spaces_secret_key
  }

  depends_on = [kubernetes_namespace.monet_app]
}

# Install NGINX Ingress Controller
resource "helm_release" "nginx_ingress" {
  name       = "nginx-ingress"
  repository = "https://kubernetes.github.io/ingress-nginx"
  chart      = "ingress-nginx"
  namespace  = "ingress-nginx"
  create_namespace = true

  set {
    name  = "controller.service.type"
    value = "LoadBalancer"
  }
  
  # Add timeout to ensure it completes
  timeout = 600
}

# Install cert-manager for SSL certificates
resource "helm_release" "cert_manager" {
  name       = "cert-manager"
  repository = "https://charts.jetstack.io"
  chart      = "cert-manager"
  namespace  = "cert-manager"
  create_namespace = true

  set {
    name  = "installCRDs"
    value = "true"
  }
  
  # Add timeout to ensure it completes
  timeout = 600
}

# Create API key secret for Unsplash if provided
resource "kubernetes_secret" "unsplash_api_key" {
  count = var.unsplash_api_key != "" ? 1 : 0
  
  metadata {
    name      = "app-secrets"
    namespace = "monet-app"
  }

  data = {
    unsplash_api_key = var.unsplash_api_key
  }
} 