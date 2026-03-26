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

# --- AMI: Amazon Linux 2023 (ARM64 for r6gd Graviton) ---

data "aws_ami" "al2023_arm64" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-arm64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "architecture"
    values = ["arm64"]
  }
}

# --- EC2 Instance ---

resource "aws_instance" "qdrant" {
  ami                    = data.aws_ami.al2023_arm64.id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.qdrant.id]

  # User data: mount NVMe, install Docker, start Qdrant
  user_data = templatefile("${path.module}/user_data.sh", {
    qdrant_version = var.qdrant_version
    qdrant_api_key = var.qdrant_api_key
  })

  root_block_device {
    volume_type = "gp3"
    volume_size = 30
    encrypted   = true
  }

  tags = {
    Name    = "${var.name_prefix}-ec2"
    Project = "card-oracle-max"
    Stage   = "1"
  }
}

# --- Optional Elastic IP ---

resource "aws_eip" "qdrant" {
  count    = var.assign_eip ? 1 : 0
  instance = aws_instance.qdrant.id
  domain   = "vpc"

  tags = {
    Name    = "${var.name_prefix}-eip"
    Project = "card-oracle-max"
  }
}
