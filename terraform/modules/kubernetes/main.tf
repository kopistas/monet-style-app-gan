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
  
  # Enable proper SSL/TLS termination
  set {
    name  = "controller.config.ssl-protocols"
    value = "TLSv1.2 TLSv1.3"
  }
  
  set {
    name  = "controller.config.ssl-ciphers"
    value = "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384"
  }
  
  set {
    name  = "controller.config.ssl-prefer-server-ciphers"
    value = "off"
  }
  
  # Disable PROXY protocol by default to allow ACME challenges to work
  set {
    name  = "controller.service.annotations.service\\.beta\\.kubernetes\\.io/do-loadbalancer-enable-proxy-protocol"
    value = "false"
  }
  
  set {
    name  = "controller.config.use-proxy-protocol"
    value = "false"
  }
  
  # Configure settings for ACME challenges
  set {
    name  = "controller.config.allow-snippet-annotations"
    value = "true"
  }
  
  set {
    name  = "controller.config.http-snippet"
    value = <<-EOT
      # Always trust traffic to /.well-known/acme-challenge/
      map $http_x_forwarded_proto $acme_redirect {
        default '';
        https '';
      }
    EOT
  }
  
  # Additional resource configuration to ensure stability
  set {
    name  = "controller.resources.requests.cpu"
    value = "100m"
  }
  
  set {
    name  = "controller.resources.requests.memory"
    value = "200Mi"
  }
  
  # Wait long enough for the load balancer to be fully provisioned
  timeout = 900
  
  depends_on = [kubernetes_namespace.monet_app]
}

# Install cert-manager for SSL certificates
resource "helm_release" "cert_manager" {
  name       = "cert-manager"
  repository = "https://charts.jetstack.io"
  chart      = "cert-manager"
  namespace  = "cert-manager"
  create_namespace = true
  version    = "v1.13.3"  # Use a specific, proven version

  set {
    name  = "installCRDs"
    value = "true"
  }
  
  # Configure resource requests/limits
  set {
    name  = "resources.requests.cpu"
    value = "100m"
  }
  
  set {
    name  = "resources.requests.memory"
    value = "300Mi"
  }
  
  set {
    name  = "resources.limits.cpu"
    value = "200m"
  }
  
  set {
    name  = "resources.limits.memory"
    value = "500Mi"
  }
  
  # Enable verbose logging for debugging
  set {
    name  = "global.logLevel"
    value = "4"
  }
  
  # Add timeout to ensure it completes
  timeout = 900
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