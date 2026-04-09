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

# ── AMI: Deep Learning AMI (PyTorch + NVIDIA drivers, Amazon Linux 2023) ──────

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

# ── Network: default VPC ──────────────────────────────────────────────────────

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ── IAM: S3 read/write for vector store and checkpoints ──────────────────────

resource "aws_iam_role" "daily_worker" {
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
    Role    = "daily-worker"
  }
}

resource "aws_iam_role_policy" "s3_access" {
  name = "${var.name_prefix}-s3-access"
  role = aws_iam_role.daily_worker.id

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

resource "aws_iam_instance_profile" "daily_worker" {
  name = "${var.name_prefix}-profile"
  role = aws_iam_role.daily_worker.name
}

# ── EC2: single g4dn.xlarge — one day of data fits on 1 GPU worker ───────────

resource "aws_instance" "daily_worker" {
  ami                    = data.aws_ami.deep_learning_gpu.id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.daily_worker.id]
  iam_instance_profile   = aws_iam_instance_profile.daily_worker.name
  subnet_id              = data.aws_subnets.default.ids[0]

  user_data = templatefile("${path.module}/user_data.sh", {
    qdrant_host      = var.qdrant_host
    qdrant_port      = var.qdrant_port
    qdrant_api_key   = var.qdrant_api_key
    s3_bucket        = var.s3_vector_bucket
    s3_prefix        = var.s3_vector_prefix
    rds_host         = var.rds_host
    rds_user         = var.rds_user
    rds_password     = var.rds_password
    rds_database     = var.rds_database
    rds2_host        = var.rds2_host
    rds2_user        = var.rds2_user
    rds2_password    = var.rds2_password
    rds2_database    = var.rds2_database
    lookback_days    = var.lookback_days
  })

  root_block_device {
    volume_type = "gp3"
    volume_size = 100
    encrypted   = true
  }

  tags = {
    Name    = "${var.name_prefix}"
    Project = "card-oracle-max"
    Role    = "daily-worker"
  }
}
