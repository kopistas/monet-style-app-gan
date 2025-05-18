terraform {
  required_providers {
    digitalocean = {
      source = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
    kubernetes = {
      source = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source = "hashicorp/helm"
      version = "~> 2.0"
    }
    time = {
      source = "hashicorp/time"
      version = "~> 0.9.1"
    }
    null = {
      source = "hashicorp/null"
      version = "~> 3.2.1"
    }
  }
  backend "s3" {
    endpoint                    = "ams3.digitaloceanspaces.com"
    region                      = "us-east-1"
    bucket                      = "terraform-storage-moria"
    key                         = "terraform-monet.tfstate"
    skip_credentials_validation = true
    skip_metadata_api_check     = true
  }
}

# Configure DigitalOcean provider at root level
provider "digitalocean" {
  token             = var.do_token
  spaces_access_id  = var.do_spaces_access_key
  spaces_secret_key = var.do_spaces_secret_key
}

# Infrastructure Module - Core Resources
module "infrastructure" {
  source = "./modules/infrastructure"
  
  do_token = var.do_token
  do_region = var.do_region
  do_spaces_region = var.do_spaces_region
  do_spaces_name = var.do_spaces_name
  do_spaces_access_key = var.do_spaces_access_key
  do_spaces_secret_key = var.do_spaces_secret_key
  cluster_node_size = var.cluster_node_size
  cluster_node_count = var.cluster_node_count
  app_domain = var.app_domain
  mlflow_domain = var.mlflow_domain
  use_do_dns = var.use_do_dns
  load_balancer_ip = var.load_balancer_ip
}

# Configure Kubernetes providers with the cluster details
provider "kubernetes" {
  host                   = module.infrastructure.kubernetes_endpoint
  token                  = module.infrastructure.kubernetes_token
  cluster_ca_certificate = base64decode(module.infrastructure.kubernetes_ca_certificate)
}

provider "helm" {
  kubernetes {
    host                   = module.infrastructure.kubernetes_endpoint
    token                  = module.infrastructure.kubernetes_token
    cluster_ca_certificate = base64decode(module.infrastructure.kubernetes_ca_certificate)
  }
}

# This null resource ensures Kubernetes providers are only used after the cluster is available
resource "null_resource" "kubernetes_config_delay" {
  depends_on = [module.infrastructure.kubernetes_cluster_id]
}

# Kubernetes Module - Only runs after cluster is available
module "kubernetes_resources" {
  source = "./modules/kubernetes"
  
  # The providers are now configured at the root level
  # so we don't need to pass kubeconfig but we use depends_on to ensure proper order
  depends_on = [module.infrastructure, null_resource.kubernetes_config_delay]
  
  do_spaces_region = var.do_spaces_region
  do_spaces_name = var.do_spaces_name
  do_spaces_access_key = var.do_spaces_access_key
  do_spaces_secret_key = var.do_spaces_secret_key
  app_domain = var.app_domain
  mlflow_domain = var.mlflow_domain
  email_for_ssl = var.email_for_ssl
  unsplash_api_key = var.unsplash_api_key
  deploy_mlflow = var.deploy_mlflow
}

# Variables
variable "do_token" {
  description = "DigitalOcean API token"
  sensitive = true
  type      = string
}

variable "do_region" {
  description = "DigitalOcean region"
  default = "ams3"
  type    = string
}

variable "do_spaces_region" {
  description = "DigitalOcean Spaces region"
  default = "ams3"
  type    = string
}

variable "do_spaces_name" {
  description = "DigitalOcean Spaces bucket name"
  default = "monet-models"
  type    = string
}

variable "do_spaces_access_key" {
  description = "DigitalOcean Spaces access key"
  sensitive = true
  type      = string
}

variable "do_spaces_secret_key" {
  description = "DigitalOcean Spaces secret key"
  sensitive = true
  type      = string
}

variable "cluster_node_size" {
  description = "Size of the Kubernetes nodes"
  default = "s-2vcpu-4gb"
  type    = string
}

variable "cluster_node_count" {
  description = "Number of nodes in the Kubernetes cluster"
  default = 2
  type    = number
}

variable "app_domain" {
  description = "Domain name for the web application"
  default = ""
  type    = string
}

variable "mlflow_domain" {
  description = "Domain name for MLflow"
  default = ""
  type    = string
}

variable "email_for_ssl" {
  description = "Email address for SSL certificates"
  default = "admin@example.com"
  type    = string
}

variable "unsplash_api_key" {
  description = "Unsplash API key for image search"
  default = ""
  sensitive = true
  type      = string
}

variable "use_do_dns" {
  description = "Whether to use DigitalOcean DNS"
  default = false
  type    = bool
}

variable "deploy_mlflow" {
  description = "Whether to deploy MLflow"
  default = false
  type    = bool
}

variable "load_balancer_ip" {
  description = "Load balancer IP for DNS records (set automatically after initial deployment)"
  default = ""
  type    = string
}

# Outputs
output "kubernetes_cluster_id" {
  value = module.infrastructure.kubernetes_cluster_id
}

output "kubernetes_endpoint" {
  value = module.infrastructure.kubernetes_endpoint
}

output "spaces_bucket_name" {
  value = module.infrastructure.spaces_bucket_name
}

output "spaces_region" {
  value = var.do_spaces_region
}

output "dns_configuration" {
  value = module.infrastructure.dns_configuration
}

output "load_balancer_ip" {
  value = var.load_balancer_ip
  description = "The IP address of the load balancer"
} 