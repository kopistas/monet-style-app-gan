terraform {
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.34.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.27.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12.1"
    }
  }
  required_version = ">= 1.0.0"

  backend "s3" {
    skip_credentials_validation = true
    skip_metadata_api_check     = true
    endpoint                    = "https://ams3.digitaloceanspaces.com"
    region                      = "us-east-1" # Required but unused for DO Spaces
    bucket                      = "terraform-storage-moria"
    key                         = "terraform-monet.tfstate"
  }
}

provider "digitalocean" {
  token = var.do_token
}

# Create a new Kubernetes cluster
resource "digitalocean_kubernetes_cluster" "monet_cluster" {
  name    = "monet-cluster"
  region  = var.do_region
  version = var.kubernetes_version

  node_pool {
    name       = "monet-nodes"
    size       = var.node_size
    node_count = var.node_count
  }
}

# Configure kubernetes provider with cluster details
provider "kubernetes" {
  host                   = digitalocean_kubernetes_cluster.monet_cluster.endpoint
  token                  = digitalocean_kubernetes_cluster.monet_cluster.kube_config[0].token
  cluster_ca_certificate = base64decode(digitalocean_kubernetes_cluster.monet_cluster.kube_config[0].cluster_ca_certificate)
}

provider "helm" {
  kubernetes {
    host                   = digitalocean_kubernetes_cluster.monet_cluster.endpoint
    token                  = digitalocean_kubernetes_cluster.monet_cluster.kube_config[0].token
    cluster_ca_certificate = base64decode(digitalocean_kubernetes_cluster.monet_cluster.kube_config[0].cluster_ca_certificate)
  }
}

# Create a Spaces bucket for model storage
resource "digitalocean_spaces_bucket" "model_storage" {
  name   = var.do_spaces_name
  region = var.do_spaces_region
  acl    = "private"
}

# Create namespaces
resource "kubernetes_namespace" "mlflow" {
  metadata {
    name = "mlflow"
  }
}

resource "kubernetes_namespace" "monet_app" {
  metadata {
    name = "monet-app"
  }
}

# Create secrets for S3 access using pre-existing DO Spaces keys
resource "kubernetes_secret" "spaces_credentials" {
  metadata {
    name      = "spaces-credentials"
    namespace = "mlflow"
  }

  data = {
    access_key = var.do_spaces_access_key
    secret_key = var.do_spaces_secret_key
  }
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
}

# Export kubeconfig for GitHub Actions
resource "local_file" "kubeconfig" {
  content  = digitalocean_kubernetes_cluster.monet_cluster.kube_config[0].raw_config
  filename = "${path.module}/kubeconfig.yaml"
}

# Automated DNS setup
# Create DNS domain if using DigitalOcean DNS
resource "digitalocean_domain" "app_domain" {
  count = var.use_do_dns ? 1 : 0
  name  = var.app_domain
}

resource "digitalocean_domain" "mlflow_domain" {
  count = var.use_do_dns ? 1 : 0
  name  = var.mlflow_domain
}

# Wait for the load balancer to be assigned an IP
resource "time_sleep" "wait_for_lb" {
  depends_on = [helm_release.nginx_ingress]
  create_duration = "60s"
}

# Create DNS records that point to the load balancer
resource "digitalocean_record" "app_record" {
  count = var.use_do_dns ? 1 : 0
  depends_on = [time_sleep.wait_for_lb]
  
  domain = digitalocean_domain.app_domain[0].name
  type   = "A"
  name   = "@"
  value  = data.kubernetes_service.ingress_nginx.status.0.load_balancer.0.ingress.0.ip
  ttl    = 300
}

resource "digitalocean_record" "mlflow_record" {
  count = var.use_do_dns ? 1 : 0
  depends_on = [time_sleep.wait_for_lb]
  
  domain = digitalocean_domain.mlflow_domain[0].name
  type   = "A"
  name   = "@"
  value  = data.kubernetes_service.ingress_nginx.status.0.load_balancer.0.ingress.0.ip
  ttl    = 300
}

# Outputs to be used by other configurations
output "kubernetes_cluster_id" {
  value = digitalocean_kubernetes_cluster.monet_cluster.id
}

output "kubernetes_endpoint" {
  value = digitalocean_kubernetes_cluster.monet_cluster.endpoint
}

output "spaces_endpoint" {
  value = "https://${var.do_spaces_name}.${var.do_spaces_region}.digitaloceanspaces.com"
}

output "spaces_bucket_name" {
  value = digitalocean_spaces_bucket.model_storage.name
}

output "spaces_region" {
  value = digitalocean_spaces_bucket.model_storage.region
}

output "load_balancer_ip" {
  value = data.kubernetes_service.ingress_nginx.status.0.load_balancer.0.ingress.0.ip
}

output "dns_configuration" {
  value = var.use_do_dns ? "DNS has been automatically configured." : "Please configure your DNS manually using the load balancer IP."
}

# Data source to get the LoadBalancer IP after it's created
data "kubernetes_service" "ingress_nginx" {
  depends_on = [helm_release.nginx_ingress]
  
  metadata {
    name      = "nginx-ingress-ingress-nginx-controller"
    namespace = "ingress-nginx"
  }
} 