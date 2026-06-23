variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-1"
}

variable "instance_type" {
  description = "CPU instance type (no GPU). c7i.2xlarge = 8 vCPU for the bulk 2025 drain; drop to c7i.xlarge (4 vCPU) for the 1-instance daily routine."
  type        = string
  default     = "c7i.2xlarge"
}

variable "worker_count" {
  description = "Number of CPU workers. 6 for the 2025 historical drain; 1 for in-service daily downloads."
  type        = number
  default     = 6
}

variable "key_pair_name" {
  description = "Existing EC2 key pair for SSH (e.g. qdrant-test)"
  type        = string
}

variable "my_ips" {
  description = "Operator IPs in CIDR notation, e.g. [\"153.53.225.179/32\"]"
  type        = list(string)
}

variable "name_prefix" {
  description = "Prefix for resource names/tags"
  type        = string
  default     = "image-archive"
}

# --- Spot (the workload is fully resumable: claim queue + orphan reaper) ---

variable "use_spot" {
  description = "Use spot instances (~70% cheaper). Safe here because days are re-queued on worker death; the reaper recovers orphaned in-flight days."
  type        = bool
  default     = true
}

variable "spot_max_price" {
  description = "Max spot price ($/hr). Empty string = cap at the on-demand price."
  type        = string
  default     = ""
}

# --- Workload config ---

variable "download_workers" {
  description = "Concurrent image downloads per worker (threads)"
  type        = number
  default     = 16
}

variable "root_volume_size" {
  description = "Root EBS size in GB. Images stream to S3, so the box needs little disk."
  type        = number
  default     = 40
}

# --- S3 ---

variable "s3_vector_bucket" {
  description = "Bucket holding the image-archive queue/markers/manifests (same as the vector bucket)"
  type        = string
}

variable "s3_image_bucket" {
  description = "Bucket for the archived images"
  type        = string
  default     = "images-130-sold"
}

# --- OpenSearch (extant read-only source) ---

variable "opensearch_host" {
  description = "Extant OpenSearch hostname (read-only source for image URLs)"
  type        = string
}

variable "opensearch_user" {
  description = "OpenSearch basic-auth username"
  type        = string
  sensitive   = true
}

variable "opensearch_password" {
  description = "OpenSearch basic-auth password"
  type        = string
  sensitive   = true
}
