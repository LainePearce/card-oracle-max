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

# ── Network: default VPC + subnets (ALB needs ≥2 AZs) ────────────────────────

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ── IAM: read-only role (S3 vector bucket for any future diagnostics) ─────────

resource "aws_iam_role" "search_worker" {
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
    Role    = "search-worker"
  }
}

resource "aws_iam_role_policy" "s3_read" {
  name = "${var.name_prefix}-s3-read"
  role = aws_iam_role.search_worker.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:ListBucket"]
      Resource = [
        "arn:aws:s3:::${var.s3_vector_bucket}",
        "arn:aws:s3:::${var.s3_vector_bucket}/*",
      ]
    }]
  })
}

resource "aws_iam_instance_profile" "search_worker" {
  name = "${var.name_prefix}-profile"
  role = aws_iam_role.search_worker.name
}

# ── EC2 instance ──────────────────────────────────────────────────────────────

resource "aws_instance" "search_worker" {
  count = var.instance_count

  ami                    = data.aws_ami.deep_learning_gpu.id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.search_worker.id]
  iam_instance_profile   = aws_iam_instance_profile.search_worker.name
  subnet_id              = data.aws_subnets.default.ids[count.index % length(data.aws_subnets.default.ids)]

  user_data = templatefile("${path.module}/user_data.sh", {
    qdrant_host        = var.qdrant_host
    qdrant_port        = var.qdrant_port
    qdrant_api_key     = var.qdrant_api_key
    opensearch_host    = var.opensearch_host
    opensearch_user    = var.opensearch_user
    opensearch_password = var.opensearch_password
    s3_vector_bucket   = var.s3_vector_bucket
    worker_port        = var.worker_port
  })

  root_block_device {
    volume_type = "gp3"
    volume_size = 100
    encrypted   = true
  }

  tags = {
    Name    = "${var.name_prefix}-${count.index}"
    Project = "card-oracle-max"
    Role    = "search-worker"
    Index   = tostring(count.index)
  }
}

# ── ALB ───────────────────────────────────────────────────────────────────────

resource "aws_lb" "search_worker" {
  name               = "${var.name_prefix}-alb"
  internal           = false   # internet-facing so local dev machines can reach it
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = data.aws_subnets.default.ids

  tags = {
    Name    = "${var.name_prefix}-alb"
    Project = "card-oracle-max"
  }
}

resource "aws_lb_target_group" "search_worker" {
  name     = "${var.name_prefix}-tg"
  port     = var.worker_port
  protocol = "HTTP"
  vpc_id   = data.aws_vpc.default.id

  health_check {
    path                = "/health"
    protocol            = "HTTP"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 10
    interval            = 30
    matcher             = "200"
  }

  tags = {
    Name    = "${var.name_prefix}-tg"
    Project = "card-oracle-max"
  }
}

resource "aws_lb_target_group_attachment" "search_worker" {
  count = var.instance_count

  target_group_arn = aws_lb_target_group.search_worker.arn
  target_id        = aws_instance.search_worker[count.index].id
  port             = var.worker_port
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.search_worker.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.search_worker.arn
  }
}
