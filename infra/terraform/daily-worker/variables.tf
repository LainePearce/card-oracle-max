variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-1"
}

variable "instance_type" {
  description = "EC2 instance type. g4dn.xlarge (T4, 16GB VRAM) is sufficient for a 1-2 day rolling window."
  type        = string
  default     = "g4dn.xlarge"
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair for SSH access"
  type        = string
}

variable "my_ips" {
  description = "CIDR blocks allowed to SSH into the worker (operator IPs only)"
  type        = list(string)
}

variable "name_prefix" {
  description = "Prefix for all resource names and tags"
  type        = string
  default     = "daily-worker"
}

# ── Qdrant ────────────────────────────────────────────────────────────────────

variable "qdrant_host" {
  description = "Qdrant host (IP or NLB DNS)"
  type        = string
}

variable "qdrant_port" {
  description = "Qdrant HTTP port"
  type        = string
  default     = "6333"
}

variable "qdrant_api_key" {
  description = "Qdrant API key"
  type        = string
  sensitive   = true
}

# ── S3 ────────────────────────────────────────────────────────────────────────

variable "s3_vector_bucket" {
  description = "S3 bucket for vector storage and daily completion markers"
  type        = string
}

variable "s3_vector_prefix" {
  description = "S3 key prefix for vectors"
  type        = string
  default     = "vectors"
}

# ── RDS primary ───────────────────────────────────────────────────────────────

variable "rds_host" {
  description = "Primary RDS MySQL hostname"
  type        = string
}

variable "rds_user" {
  description = "Primary RDS MySQL username"
  type        = string
  sensitive   = true
}

variable "rds_password" {
  description = "Primary RDS MySQL password"
  type        = string
  sensitive   = true
}

variable "rds_database" {
  description = "Primary RDS MySQL database name"
  type        = string
}

# ── RDS secondary (optional) ──────────────────────────────────────────────────

variable "rds2_host" {
  description = "Secondary RDS MySQL hostname — empty to disable"
  type        = string
  default     = ""
}

variable "rds2_user" {
  description = "Secondary RDS MySQL username"
  type        = string
  sensitive   = true
  default     = ""
}

variable "rds2_password" {
  description = "Secondary RDS MySQL password"
  type        = string
  sensitive   = true
  default     = ""
}

variable "rds2_database" {
  description = "Secondary RDS MySQL database name"
  type        = string
  default     = ""
}

# ── Daily update config ───────────────────────────────────────────────────────

variable "lookback_days" {
  description = "Number of past days the daily update processes each run. 2 catches late-arriving RDS data."
  type        = number
  default     = 2
}
