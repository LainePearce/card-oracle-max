variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-1"
}

variable "instance_type" {
  description = "EC2 instance type - g4dn.xlarge has 1x T4 GPU (16GB VRAM), 4 vCPU, 16GB RAM"
  type        = string
  default     = "g4dn.xlarge"
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair for SSH access"
  type        = string
}

variable "my_ips" {
  description = "List of operator IP addresses in CIDR notation (e.g. [\"1.2.3.4/32\", \"5.6.7.8/32\"])"
  type        = list(string)
}

variable "name_prefix" {
  description = "Prefix for resource names and tags"
  type        = string
  default     = "gpu-worker"
}

# --- Qdrant connection (points to qdrant-single instance) ---

variable "qdrant_host" {
  description = "Qdrant instance IP or hostname"
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

# --- S3 vector store ---

variable "s3_vector_bucket" {
  description = "S3 bucket name for vector storage"
  type        = string
}

variable "s3_vector_prefix" {
  description = "S3 key prefix for vectors"
  type        = string
  default     = "vectors"
}

# --- OpenSearch (extant read-only source — may be unavailable) ---

variable "opensearch_host" {
  description = "Extant OpenSearch cluster hostname (read-only source, optional)"
  type        = string
  default     = ""
}

variable "opensearch_user" {
  description = "Extant OpenSearch basic auth username"
  type        = string
  sensitive   = true
  default     = ""
}

variable "opensearch_password" {
  description = "Extant OpenSearch basic auth password"
  type        = string
  sensitive   = true
  default     = ""
}

# --- New self-managed OpenSearch (write target) ---

variable "opensearch_docs_host" {
  description = "New OpenSearch cluster NLB DNS (without http://) -- Stage 2 only; leave empty for Stage 1 backfill"
  type        = string
  default     = ""
}

variable "opensearch_docs_user" {
  description = "New OpenSearch admin username"
  type        = string
  sensitive   = true
  default     = "admin"
}

variable "opensearch_docs_password" {
  description = "New OpenSearch admin password -- Stage 2 only; leave empty for Stage 1 backfill"
  type        = string
  sensitive   = true
  default     = ""
}

# --- RDS primary (current data source) ---

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

# --- RDS secondary (older DB — queried alongside primary for gap coverage) ---
# Leave rds2_host empty to disable (primary-only mode).

variable "rds2_host" {
  description = "Secondary (older) RDS MySQL hostname — empty to disable"
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

# --- Parallel backfill partitions ---

variable "worker_count" {
  description = "Number of GPU workers to provision. Phase date splits are handled by tools/worker_phases.py."
  type        = number
  default     = 12
}
