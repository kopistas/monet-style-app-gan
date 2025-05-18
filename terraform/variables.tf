variable "do_token" {
  description = "DigitalOcean API token"
  type        = string
  sensitive   = true
}

variable "do_region" {
  description = "DigitalOcean region"
  type        = string
  default     = "fra1"
}

variable "do_spaces_region" {
  description = "DigitalOcean Spaces region"
  type        = string
  default     = "fra1"
}

variable "do_spaces_name" {
  description = "DigitalOcean Spaces bucket name"
  type        = string
  default     = "monet-models"
}

variable "do_spaces_access_key" {
  description = "DigitalOcean Spaces access key"
  type        = string
  sensitive   = true
}

variable "do_spaces_secret_key" {
  description = "DigitalOcean Spaces secret key"
  type        = string
  sensitive   = true
}

variable "kubernetes_version" {
  description = "Kubernetes version to use"
  type        = string
  default     = "1.29.1-do.0"
}

variable "node_size" {
  description = "Droplet size for Kubernetes nodes"
  type        = string
  default     = "s-2vcpu-4gb"
}

variable "node_count" {
  description = "Number of nodes in the Kubernetes cluster"
  type        = number
  default     = 2
}

variable "app_domain" {
  description = "Domain for the Monet app"
  type        = string
  default     = "monet-app.example.com"
}

variable "mlflow_domain" {
  description = "Domain for MLflow"
  type        = string
  default     = "mlflow.example.com"
}

variable "unsplash_api_key" {
  description = "API key for Unsplash"
  type        = string
  sensitive   = true
  default     = ""
}

variable "email_for_ssl" {
  description = "Email for Let's Encrypt certificates"
  type        = string
  default     = "admin@example.com"
}

variable "use_do_dns" {
  description = "Whether to use DigitalOcean DNS for domain configuration"
  type        = bool
  default     = false
} 