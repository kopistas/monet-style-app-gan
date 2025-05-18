variable "do_token" {
  description = "DigitalOcean API token"
  sensitive = true
}

variable "do_region" {
  description = "DigitalOcean region"
  default = "nyc1"
}

variable "do_spaces_region" {
  description = "DigitalOcean Spaces region"
  default = "nyc3"
}

variable "do_spaces_name" {
  description = "DigitalOcean Spaces bucket name"
  default = "monet-models"
}

variable "do_spaces_access_key" {
  description = "DigitalOcean Spaces access key"
  sensitive = true
}

variable "do_spaces_secret_key" {
  description = "DigitalOcean Spaces secret key"
  sensitive = true
}

variable "cluster_node_size" {
  description = "Size of the Kubernetes nodes"
  default = "s-2vcpu-4gb"
}

variable "cluster_node_count" {
  description = "Number of nodes in the Kubernetes cluster"
  default = 2
}

variable "app_domain" {
  description = "Domain name for the web application"
  default = ""
}

variable "mlflow_domain" {
  description = "Domain name for MLflow"
  default = ""
}

variable "use_do_dns" {
  description = "Whether to use DigitalOcean DNS"
  default = false
} 