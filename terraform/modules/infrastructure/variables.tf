variable "do_token" {
  description = "DigitalOcean API token"
  sensitive = true
  type        = string
}

variable "do_region" {
  description = "DigitalOcean region"
  default = "nyc1"
  type    = string
}

variable "do_spaces_region" {
  description = "DigitalOcean Spaces region"
  default = "nyc3"
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

variable "use_do_dns" {
  description = "Whether to use DigitalOcean DNS"
  default = false
  type    = bool
} 