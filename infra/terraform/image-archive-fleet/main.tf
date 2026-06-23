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

# --- AMI: stock Amazon Linux 2023 x86_64 (no GPU/CUDA — image archival is CPU/IO only) ---

data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# --- IAM: read/write the image bucket + the queue bucket; no Qdrant/RDS needed ---

resource "aws_iam_role" "worker" {
  name = "${var.name_prefix}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = { Project = "card-oracle-max" }
}

resource "aws_iam_role_policy" "s3_access" {
  name = "${var.name_prefix}-s3-access"
  role = aws_iam_role.worker.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["s3:PutObject", "s3:GetObject", "s3:ListBucket", "s3:DeleteObject"]
      Resource = [
        "arn:aws:s3:::${var.s3_vector_bucket}",
        "arn:aws:s3:::${var.s3_vector_bucket}/*",
        "arn:aws:s3:::${var.s3_image_bucket}",
        "arn:aws:s3:::${var.s3_image_bucket}/*",
      ]
    }]
  })
}

resource "aws_iam_instance_profile" "worker" {
  name = "${var.name_prefix}-profile"
  role = aws_iam_role.worker.name
}

# --- EC2 instances ---

resource "aws_instance" "worker" {
  count = var.worker_count

  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.worker.id]
  iam_instance_profile   = aws_iam_instance_profile.worker.name

  # Spot when enabled — interruptions are safe: the day re-queues and the reaper
  # recovers it. Note: with count-based instances a reclaimed spot box is not
  # auto-replaced; re-run `apply` to top the fleet back up.
  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        max_price                      = var.spot_max_price != "" ? var.spot_max_price : null
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
      }
    }
  }

  user_data = templatefile("${path.module}/user_data.sh", {
    s3_vector_bucket = var.s3_vector_bucket
    s3_image_bucket  = var.s3_image_bucket
    os_host          = var.opensearch_host
    os_user          = var.opensearch_user
    os_password      = var.opensearch_password
    download_workers = var.download_workers
    worker_index     = count.index
  })

  root_block_device {
    volume_type = "gp3"
    volume_size = var.root_volume_size
    encrypted   = true
  }

  tags = {
    Name    = "${var.name_prefix}-worker-${count.index}"
    Project = "card-oracle-max"
    Role    = "image-archive"
    Worker  = tostring(count.index)
  }
}
