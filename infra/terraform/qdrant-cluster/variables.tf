variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-1"
}

variable "instance_type" {
  description = "EC2 instance type — r6gd.8xlarge: 256GB RAM, 1.9TB NVMe, 32 vCPU"
  type        = string
  default     = "r6gd.8xlarge"
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair for SSH access"
  type        = string
}

variable "qdrant_api_key" {
  description = "API key for Qdrant REST/gRPC authentication"
  type        = string
  sensitive   = true
}

variable "my_ip" {
  description = "Your IP in CIDR notation for SSH + API access (e.g. 1.2.3.4/32)"
  type        = string
}

variable "qdrant_version" {
  description = "Qdrant Docker image tag"
  type        = string
  default     = "v1.8.2"
}

variable "name_prefix" {
  description = "Prefix for resource names and tags"
  type        = string
  default     = "qdrant-cluster"
}

variable "shard_number" {
  description = "Number of shards for the cards collection (2 per node)"
  type        = number
  default     = 6
}

variable "replication_factor" {
  description = "Replication factor for the cards collection"
  type        = number
  default     = 2
}
