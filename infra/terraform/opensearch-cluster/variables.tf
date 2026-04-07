variable "aws_region" {
  type    = string
  default = "us-west-1"
}

variable "cluster_name" {
  description = "Name prefix for all cluster resources"
  type        = string
  default     = "card-oracle-es"
}

variable "elasticsearch_version" {
  description = "Elasticsearch version to install (e.g. 8.13.4)"
  type        = string
  default     = "8.13.4"
}

variable "key_pair_name" {
  description = "EC2 key pair for SSH access"
  type        = string
}

# ── Instance types (matching managed service) ────────────────────────────────

variable "data_instance_type" {
  description = "Data node — c7g.2xlarge: 8 vCPU, 16GB RAM (equiv to c7g.2xlarge.search)"
  type        = string
  default     = "c7g.2xlarge"
}

variable "master_instance_type" {
  description = "Master node — r7g.xlarge: 4 vCPU, 32GB RAM (equiv to r7g.xlarge.search)"
  type        = string
  default     = "r7g.xlarge"
}

# ── JVM heap (half of instance RAM) ──────────────────────────────────────────

variable "data_jvm_heap" {
  description = "JVM heap for data nodes — half of 16GB RAM"
  type        = string
  default     = "8g"
}

variable "master_jvm_heap" {
  description = "JVM heap for master nodes — half of 32GB RAM"
  type        = string
  default     = "16g"
}

# ── EBS (matching managed service) ───────────────────────────────────────────

variable "data_volume_gb" {
  description = "EBS volume size per data node (GiB)"
  type        = number
  default     = 100
}

variable "data_volume_iops" {
  description = "EBS provisioned IOPS per data node"
  type        = number
  default     = 4500
}

variable "data_volume_throughput" {
  description = "EBS throughput per data node (MiB/s)"
  type        = number
  default     = 250
}

# ── Networking ────────────────────────────────────────────────────────────────

variable "vpc_id" {
  description = "VPC to deploy into"
  type        = string
}

variable "subnet_ids" {
  description = "One subnet per AZ for cluster nodes (3 required)"
  type        = list(string)
  validation {
    condition     = length(var.subnet_ids) == 3
    error_message = "Exactly 3 subnet IDs required (one per AZ)."
  }
}

variable "subnet_cidrs" {
  description = "CIDR blocks of the 3 subnets — used to assign static master IPs"
  type        = list(string)
  validation {
    condition     = length(var.subnet_cidrs) == 3
    error_message = "Exactly 3 subnet CIDRs required."
  }
}

variable "operator_ips" {
  description = "CIDR list of operator IPs allowed SSH + HTTP access"
  type        = list(string)
}

variable "client_security_group_ids" {
  description = "SG IDs of Lambda / GPU worker that need port 9200 access"
  type        = list(string)
  default     = []
}

variable "internal_nlb" {
  description = "Set true for VPC-internal NLB (recommended for production)"
  type        = bool
  default     = false
}

# ── Auth + snapshots ──────────────────────────────────────────────────────────

variable "admin_password" {
  description = "OpenSearch admin user password"
  type        = string
  sensitive   = true
}

variable "snapshot_bucket" {
  description = "S3 bucket for automated OpenSearch snapshots"
  type        = string
}
