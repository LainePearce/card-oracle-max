terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# --- AMI: Deep Learning AMI with PyTorch + NVIDIA drivers (AL2023 x86_64) ---

data "aws_ami" "deep_learning_gpu" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Amazon Linux 2023) *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# --- IAM Role: S3 + OpenSearch access ---

resource "aws_iam_role" "gpu_worker" {
  name = "${var.name_prefix}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = {
    Project = "card-oracle-max"
    Stage   = "1"
  }
}

resource "aws_iam_role_policy" "s3_access" {
  name = "${var.name_prefix}-s3-access"
  role = aws_iam_role.gpu_worker.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket",
        "s3:DeleteObject",
      ]
      Resource = [
        "arn:aws:s3:::${var.s3_vector_bucket}",
        "arn:aws:s3:::${var.s3_vector_bucket}/*",
      ]
    }]
  })
}

resource "aws_iam_instance_profile" "gpu_worker" {
  name = "${var.name_prefix}-profile"
  role = aws_iam_role.gpu_worker.name
}

# --- EC2 Instances (one per worker) ---
# Phase date splits are computed in tools/worker_phases.py, not Terraform.

resource "aws_instance" "gpu_worker" {
  count = var.worker_count

  ami                    = data.aws_ami.deep_learning_gpu.id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.gpu_worker.id]
  iam_instance_profile   = aws_iam_instance_profile.gpu_worker.name

  user_data = templatefile("${path.module}/user_data.sh", {
    qdrant_host      = var.qdrant_host
    qdrant_port      = var.qdrant_port
    qdrant_api_key   = var.qdrant_api_key
    s3_bucket        = var.s3_vector_bucket
    s3_prefix        = var.s3_vector_prefix
    os_host          = var.opensearch_host
    os_user          = var.opensearch_user
    os_password      = var.opensearch_password
    os_docs_host     = var.opensearch_docs_host
    os_docs_user     = var.opensearch_docs_user
    os_docs_password = var.opensearch_docs_password
    worker_index     = count.index
    rds_host         = var.rds_host
    rds_user         = var.rds_user
    rds_password     = var.rds_password
    rds_database     = var.rds_database
    rds2_host        = var.rds2_host
    rds2_user        = var.rds2_user
    rds2_password    = var.rds2_password
    rds2_database    = var.rds2_database
  })

  root_block_device {
    volume_type = "gp3"
    volume_size = 100
    encrypted   = true
  }

  tags = {
    Name    = "${var.name_prefix}-worker-${count.index}"
    Project = "card-oracle-max"
    Stage   = "2"
    Worker  = tostring(count.index)
  }
}
