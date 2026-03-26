variable "aws_region" {
  description = "AWS region for the Qdrant test instance"
  type        = string
  default     = "us-west-1"
}

variable "instance_type" {
  description = "EC2 instance type — r6gd.xlarge has 32GB RAM + 118GB NVMe"
  type        = string
  default     = "r6gd.xlarge"
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair for SSH access"
  type        = string
}

variable "qdrant_api_key" {
  description = "API key for Qdrant authentication"
  type        = string
  sensitive   = true
}

variable "my_ip" {
  description = "Your IP address in CIDR notation for security group ingress (e.g. 1.2.3.4/32)"
  type        = string
}

variable "qdrant_version" {
  description = "Qdrant Docker image tag"
  type        = string
  default     = "v1.8.2"
}

variable "assign_eip" {
  description = "Whether to assign an Elastic IP for a stable public address"
  type        = bool
  default     = false
}

variable "name_prefix" {
  description = "Prefix for resource names and tags"
  type        = string
  default     = "qdrant-test"
}
