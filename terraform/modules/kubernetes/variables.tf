variable "do_spaces_region" {
  description = "DigitalOcean Spaces region"
}

variable "do_spaces_name" {
  description = "DigitalOcean Spaces bucket name" 
}

variable "do_spaces_access_key" {
  description = "DigitalOcean Spaces access key"
  sensitive = true
}

variable "do_spaces_secret_key" {
  description = "DigitalOcean Spaces secret key"
  sensitive = true
}

variable "app_domain" {
  description = "Domain name for the web application"
  default = ""
}

variable "mlflow_domain" {
  description = "Domain name for MLflow"
  default = ""
}

variable "email_for_ssl" {
  description = "Email address for SSL certificates"
  default = "admin@example.com"
}

variable "unsplash_api_key" {
  description = "Unsplash API key for image search"
  default = ""
}

variable "mlflow_username" {
  description = "Username for MLflow authentication"
  default = ""
  sensitive = true
}

variable "mlflow_password" {
  description = "Password for MLflow authentication"
  default = ""
  sensitive = true
}

variable "deploy_mlflow" {
  description = "Whether to deploy MLflow"
  default = true
  type    = bool
} 