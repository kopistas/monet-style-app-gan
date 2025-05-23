# Infrastructure Module for DigitalOcean Resources

# Configure DigitalOcean provider
provider "digitalocean" {
  token             = var.do_token
  spaces_access_id  = var.do_spaces_access_key
  spaces_secret_key = var.do_spaces_secret_key
}

# Create a new Kubernetes cluster
resource "digitalocean_kubernetes_cluster" "monet_cluster" {
  name    = "monet-cluster"
  region  = var.do_region
  version = "1.32.2-do.1"
  
  node_pool {
    name       = "monet-nodes"
    size       = var.cluster_node_size
    node_count = var.cluster_node_count
  }
}

# Sleep to allow the load balancer time to initialize
resource "time_sleep" "wait_for_lb" {
  depends_on = [digitalocean_kubernetes_cluster.monet_cluster]
  create_duration = "60s"
}

# Create a Spaces bucket for model storage
resource "digitalocean_spaces_bucket" "model_storage" {
  name   = var.do_spaces_name
  region = var.do_spaces_region
  acl    = "private"
  force_destroy = true
}

# Export kubeconfig for other modules and GitHub Actions
resource "local_file" "kubeconfig" {
  content  = digitalocean_kubernetes_cluster.monet_cluster.kube_config[0].raw_config
  filename = "${path.module}/../../kubeconfig.yaml"
}

# Create DNS domains if using DigitalOcean DNS
resource "digitalocean_domain" "app_domain" {
  count = var.use_do_dns && var.app_domain != "" ? 1 : 0
  name  = var.app_domain
}

resource "digitalocean_domain" "mlflow_domain" {
  count = var.use_do_dns && var.mlflow_domain != "" ? 1 : 0
  name  = var.mlflow_domain
}

# Create DNS records for app and MLflow domains
# We will use the load balancer IP from the var.load_balancer_ip 
# which will be updated by the workflow
resource "digitalocean_record" "app_record" {
  count  = var.use_do_dns && var.app_domain != "" ? 1 : 0
  domain = digitalocean_domain.app_domain[0].name
  type   = "A"
  name   = "@"
  value  = var.load_balancer_ip != "" ? var.load_balancer_ip : "127.0.0.1"
  ttl    = 300
}

resource "digitalocean_record" "mlflow_record" {
  count  = var.use_do_dns && var.mlflow_domain != "" ? 1 : 0
  domain = digitalocean_domain.mlflow_domain[0].name
  type   = "A"
  name   = "@"
  value  = var.load_balancer_ip != "" ? var.load_balancer_ip : "127.0.0.1"
  ttl    = 300
}

resource "digitalocean_record" "app_record_www" {
  count  = var.use_do_dns && var.app_domain != "" ? 1 : 0
  domain = digitalocean_domain.app_domain[0].name
  type   = "A"
  name   = "www"
  value  = var.load_balancer_ip != "" ? var.load_balancer_ip : "127.0.0.1"
  ttl    = 300
}

# Outputs
output "kubernetes_cluster_id" {
  value = digitalocean_kubernetes_cluster.monet_cluster.id
}

output "kubernetes_endpoint" {
  value = digitalocean_kubernetes_cluster.monet_cluster.endpoint
}

output "kubernetes_token" {
  value     = digitalocean_kubernetes_cluster.monet_cluster.kube_config[0].token
  sensitive = true
}

output "kubernetes_ca_certificate" {
  value     = digitalocean_kubernetes_cluster.monet_cluster.kube_config[0].cluster_ca_certificate
  sensitive = true
}

output "kubeconfig" {
  value = digitalocean_kubernetes_cluster.monet_cluster.kube_config[0].raw_config
  sensitive = true
}

output "spaces_bucket_name" {
  value = digitalocean_spaces_bucket.model_storage.name
}

output "dns_configuration" {
  value = var.use_do_dns ? "DNS has been automatically configured." : "Please configure your DNS manually."
} 