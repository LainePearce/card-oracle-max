variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-1"
}

variable "instance_types" {
  description = "CPU instance types for the ASG to diversify across (spot resilience). All ~8 vCPU, all available in us-west-1 (older region — no c7a/m7a, limited c7i AZs). First is the launch-template default."
  type        = list(string)
  default     = ["c7i.2xlarge", "c6i.2xlarge", "m6i.2xlarge", "c5.2xlarge", "m5.2xlarge"]
}

variable "worker_count" {
  description = "Desired ASG size (= min). 1 for in-service hourly incremental downloads; bump to 6 for a historical re-drain."
  type        = number
  default     = 1
}

variable "on_demand_base_capacity" {
  description = "Instances guaranteed on-demand (the rest are spot). 0 = all spot (cheapest); set 1 to keep one stable anchor."
  type        = number
  default     = 0
}

variable "code_s3_key" {
  description = "S3 key (under the vector bucket) of the code tarball that user_data fetches at boot. Publish with infra/scripts/publish-image-archive-code.sh."
  type        = string
  default     = "deploy/image-archive-code.tar.gz"
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
