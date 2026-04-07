variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-1"
}

variable "instance_type" {
  description = "EC2 instance type — g4dn.xlarge (T4, 16GB VRAM) for initial testing; g5.xlarge (A10G, 24GB) for production"
  type        = string
  default     = "g4dn.xlarge"
}

variable "instance_count" {
  description = "Number of GPU search worker instances. Start with 1 to validate, increase once stable."
  type        = number
  default     = 1
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair for SSH access"
  type        = string
}

variable "my_ips" {
  description = "CIDR blocks allowed to access ALB (port 80) and SSH (port 22). Use [\"0.0.0.0/0\"] to open to internet."
  type        = list(string)
}

variable "name_prefix" {
  description = "Prefix for all resource names and tags"
  type        = string
  default     = "search-worker"
}

variable "worker_port" {
  description = "Port the gpu_worker_server.py Gunicorn process listens on"
  type        = number
  default     = 8081
}

# ── Qdrant ────────────────────────────────────────────────────────────────────

variable "qdrant_host" {
  description = "Qdrant host (IP or NLB DNS). The search worker connects to this for ANN queries."
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

# ── OpenSearch (result enrichment) ────────────────────────────────────────────

variable "opensearch_host" {
  description = "OpenSearch cluster hostname for document enrichment (no https:// prefix)"
  type        = string
  default     = ""
}

variable "opensearch_user" {
  description = "OpenSearch basic auth username"
  type        = string
  sensitive   = true
  default     = ""
}

variable "opensearch_password" {
  description = "OpenSearch basic auth password"
  type        = string
  sensitive   = true
  default     = ""
}

# ── S3 ────────────────────────────────────────────────────────────────────────

variable "s3_vector_bucket" {
  description = "S3 bucket for vector storage (read-only access granted for diagnostics)"
  type        = string
}
