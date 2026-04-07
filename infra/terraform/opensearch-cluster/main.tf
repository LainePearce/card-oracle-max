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

# ── AZs ──────────────────────────────────────────────────────────────────────

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  azs = slice(data.aws_availability_zones.available.names, 0, 3)

  # Master node private IPs are assigned statically so data nodes can reference
  # them at bootstrap time without a circular Terraform dependency.
  master_private_ips = [
    cidrhost(var.subnet_cidrs[0], 10),
    cidrhost(var.subnet_cidrs[1], 10),
    cidrhost(var.subnet_cidrs[2], 10),
  ]

  master_ip_list   = join(",", [for ip in local.master_private_ips : "\"${ip}\""])
  master_node_names = [for i in range(3) : "master-${i}"]
  master_names_list = join(",", [for n in local.master_node_names : "\"${n}\""])
}

# ── AMI: Amazon Linux 2023 (ARM64 for c7g/r7g Graviton3) ─────────────────────

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
}

# ── VPC / Subnets (use existing or pass in via variables) ─────────────────────

data "aws_vpc" "selected" {
  id = var.vpc_id
}

data "aws_subnet" "cluster" {
  count = 3
  id    = var.subnet_ids[count.index]
}

# ── IAM role (for CloudWatch logs + Secrets access) ──────────────────────────

resource "aws_iam_role" "opensearch_node" {
  name = "${var.cluster_name}-node-role"

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

resource "aws_iam_role_policy" "opensearch_node" {
  name = "${var.cluster_name}-node-policy"
  role = aws_iam_role.opensearch_node.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # CloudWatch logs
        Effect = "Allow"
        Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "*"
      },
      {
        # S3 snapshots bucket
        Effect   = "Allow"
        Action   = ["s3:PutObject", "s3:GetObject", "s3:DeleteObject", "s3:ListBucket"]
        Resource = [
          "arn:aws:s3:::${var.snapshot_bucket}",
          "arn:aws:s3:::${var.snapshot_bucket}/*",
        ]
      },
    ]
  })
}

resource "aws_iam_instance_profile" "opensearch_node" {
  name = "${var.cluster_name}-node-profile"
  role = aws_iam_role.opensearch_node.name
}

# ── EBS volumes for data nodes ────────────────────────────────────────────────

resource "aws_ebs_volume" "data" {
  count             = 3
  availability_zone = local.azs[count.index]
  type              = "gp3"
  size              = var.data_volume_gb        # 100 GB
  iops              = var.data_volume_iops       # 4500
  throughput        = var.data_volume_throughput # 250 MiB/s
  encrypted         = true

  tags = {
    Name    = "${var.cluster_name}-data-${count.index}-vol"
    Project = "card-oracle-max"
  }
}

# ── Master nodes (r7g.xlarge — dedicated, no data) ───────────────────────────

resource "aws_instance" "master" {
  count = 3

  ami                    = data.aws_ami.al2023_arm64.id
  instance_type          = var.master_instance_type   # r7g.xlarge
  key_name               = var.key_pair_name
  subnet_id              = var.subnet_ids[count.index]
  vpc_security_group_ids = [aws_security_group.opensearch_cluster.id, aws_security_group.opensearch_client.id]
  iam_instance_profile   = aws_iam_instance_profile.opensearch_node.name
  private_ip             = local.master_private_ips[count.index]

  user_data = templatefile("${path.module}/user_data_master.sh", {
    cluster_name        = var.cluster_name
    node_name           = "master-${count.index}"
    node_index          = count.index
    elasticsearch_version  = var.elasticsearch_version
    master_ip_list      = local.master_ip_list
    master_names_list   = local.master_names_list
    admin_password      = var.admin_password
    jvm_heap            = var.master_jvm_heap  # "16g"
    aws_region          = var.aws_region
    snapshot_bucket     = var.snapshot_bucket
  })

  root_block_device {
    volume_type = "gp3"
    volume_size = 30
    encrypted   = true
  }

  tags = {
    Name             = "${var.cluster_name}-master-${count.index}"
    Project          = "card-oracle-max"
    ClusterName      = var.cluster_name
    ElasticsearchRole   = "master"
  }
}

# ── Data nodes (c7g.2xlarge — data + ingest) ─────────────────────────────────

resource "aws_instance" "data" {
  count = 3

  ami                    = data.aws_ami.al2023_arm64.id
  instance_type          = var.data_instance_type   # c7g.2xlarge
  key_name               = var.key_pair_name
  subnet_id              = var.subnet_ids[count.index]
  vpc_security_group_ids = [aws_security_group.opensearch_cluster.id, aws_security_group.opensearch_client.id]
  iam_instance_profile   = aws_iam_instance_profile.opensearch_node.name

  user_data = templatefile("${path.module}/user_data_data.sh", {
    cluster_name        = var.cluster_name
    node_name           = "data-${count.index}"
    node_index          = count.index
    elasticsearch_version  = var.elasticsearch_version
    master_ip_list      = local.master_ip_list
    master_names_list   = local.master_names_list
    admin_password      = var.admin_password
    jvm_heap            = var.data_jvm_heap    # "8g"
    aws_region          = var.aws_region
    snapshot_bucket     = var.snapshot_bucket
    ebs_device          = "/dev/xvdf"
  })

  root_block_device {
    volume_type = "gp3"
    volume_size = 30
    encrypted   = true
  }

  depends_on = [aws_instance.master]

  tags = {
    Name             = "${var.cluster_name}-data-${count.index}"
    Project          = "card-oracle-max"
    ClusterName      = var.cluster_name
    ElasticsearchRole   = "data"
  }
}

# ── Attach EBS volumes to data nodes ─────────────────────────────────────────

resource "aws_volume_attachment" "data" {
  count       = 3
  device_name = "/dev/xvdf"
  volume_id   = aws_ebs_volume.data[count.index].id
  instance_id = aws_instance.data[count.index].id
}

# ── Network Load Balancer (port 9200 → data nodes) ───────────────────────────

resource "aws_lb" "opensearch" {
  name               = "${var.cluster_name}-nlb"
  load_balancer_type = "network"
  internal           = var.internal_nlb
  subnets            = var.subnet_ids

  tags = { Project = "card-oracle-max" }
}

resource "aws_lb_target_group" "opensearch_http" {
  name        = "${var.cluster_name}-http"
  port        = 9200
  protocol    = "TCP"
  vpc_id      = var.vpc_id
  target_type = "instance"

  health_check {
    protocol            = "HTTP"
    path                = "/_cluster/health"
    port                = "9200"
    healthy_threshold   = 2
    unhealthy_threshold = 2
    interval            = 15
  }
}

resource "aws_lb_listener" "opensearch_http" {
  load_balancer_arn = aws_lb.opensearch.arn
  port              = 9200
  protocol          = "TCP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.opensearch_http.arn
  }
}

resource "aws_lb_target_group_attachment" "data" {
  count            = 3
  target_group_arn = aws_lb_target_group.opensearch_http.arn
  target_id        = aws_instance.data[count.index].id
  port             = 9200
}
