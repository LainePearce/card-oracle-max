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

# --- AMI: stock Amazon Linux 2023 x86_64 (no GPU — archival is CPU/IO only) ---

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

# --- Default VPC subnets (ASG spreads across AZs for spot resilience) ---

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# --- IAM: read/write the image bucket + the queue bucket (incl. the code tarball) ---

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

# --- Launch template: self-bootstrapping (fetches code tarball from S3) ---

resource "aws_launch_template" "worker" {
  name_prefix   = "${var.name_prefix}-lt-"
  image_id      = data.aws_ami.al2023.id
  instance_type = var.instance_types[0] # overridden per-type by the ASG policy
  key_name      = var.key_pair_name

  vpc_security_group_ids = [aws_security_group.worker.id]

  iam_instance_profile {
    name = aws_iam_instance_profile.worker.name
  }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size = var.root_volume_size
      volume_type = "gp3"
      encrypted   = true
    }
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    s3_vector_bucket = var.s3_vector_bucket
    s3_image_bucket  = var.s3_image_bucket
    code_key         = var.code_s3_key
    os_host          = var.opensearch_host
    os_user          = var.opensearch_user
    os_password      = var.opensearch_password
    download_workers = var.download_workers
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name    = "${var.name_prefix}-worker"
      Project = "card-oracle-max"
      Role    = "image-archive"
    }
  }
}

# --- ASG: diversified spot + capacity rebalancing = auto-replacement ---

resource "aws_autoscaling_group" "worker" {
  name                = "${var.name_prefix}-asg"
  vpc_zone_identifier = data.aws_subnets.default.ids

  desired_capacity = var.worker_count
  min_size         = var.worker_count
  max_size         = var.worker_count + 2

  # Proactively replace instances AWS flags for spot interruption.
  capacity_rebalance = true
  health_check_type  = "EC2"

  mixed_instances_policy {
    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.worker.id
        version            = "$Latest"
      }
      # Diversify across instance types so a reclaim of one type is replaced
      # from another type's spot pool — the key to spot resilience.
      dynamic "override" {
        for_each = var.instance_types
        content {
          instance_type = override.value
        }
      }
    }
    instances_distribution {
      on_demand_base_capacity                  = var.on_demand_base_capacity
      on_demand_percentage_above_base_capacity = 0
      spot_allocation_strategy                 = "price-capacity-optimized"
    }
  }

  tag {
    key                 = "Name"
    value               = "${var.name_prefix}-worker"
    propagate_at_launch = true
  }

  # Replace instances when the launch template (e.g. new code_key) changes.
  instance_refresh {
    strategy = "Rolling"
    preferences {
      min_healthy_percentage = 50
    }
  }
}
