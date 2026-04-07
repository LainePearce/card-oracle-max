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

# ── AMI: Amazon Linux 2023 ARM64 (for r6gd Graviton) ─────────────────────────

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

# ── Subnets (NLB needs at least one; use all available in default VPC) ────────

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ── Node 0 — Seed (initialises the cluster) ───────────────────────────────────
#
# Starts standalone — other nodes bootstrap from its private IP.
# Created first so peer nodes can reference its private_ip.

resource "aws_instance" "qdrant_seed" {
  ami                    = data.aws_ami.al2023_arm64.id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.qdrant_cluster.id]

  user_data = templatefile("${path.module}/user_data_seed.sh", {
    qdrant_version = var.qdrant_version
    qdrant_api_key = var.qdrant_api_key
  })

  root_block_device {
    volume_type = "gp3"
    volume_size = 30
    encrypted   = true
  }

  tags = {
    Name    = "${var.name_prefix}-node-0"
    Role    = "seed"
    Project = "card-oracle-max"
    Stage   = "1"
  }
}

# ── Nodes 1 & 2 — Peers (join via seed) ──────────────────────────────────────
#
# Launched after seed so its private IP is known.
# Each runs Qdrant with --bootstrap pointing at the seed node's P2P port.

resource "aws_instance" "qdrant_peer" {
  count                  = 2
  ami                    = data.aws_ami.al2023_arm64.id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.qdrant_cluster.id]

  user_data = templatefile("${path.module}/user_data_peer.sh", {
    qdrant_version  = var.qdrant_version
    qdrant_api_key  = var.qdrant_api_key
    seed_private_ip = aws_instance.qdrant_seed.private_ip
  })

  root_block_device {
    volume_type = "gp3"
    volume_size = 30
    encrypted   = true
  }

  depends_on = [aws_instance.qdrant_seed]

  tags = {
    Name    = "${var.name_prefix}-node-${count.index + 1}"
    Role    = "peer"
    Project = "card-oracle-max"
    Stage   = "1"
  }
}

# ── Network Load Balancer ─────────────────────────────────────────────────────
#
# Internet-facing NLB with TCP passthrough — required for gRPC (HTTP/2).
# Security on the nodes is enforced by the instance security group.
# NLBs do not have their own security groups.

resource "aws_lb" "qdrant" {
  name               = "${var.name_prefix}-nlb"
  internal           = true   # VPC-internal: avoids hairpinning when accessed from within the VPC
  load_balancer_type = "network"
  subnets            = data.aws_subnets.default.ids

  enable_cross_zone_load_balancing = true
  enable_deletion_protection       = false

  tags = {
    Name    = "${var.name_prefix}-nlb"
    Project = "card-oracle-max"
  }
}

# ── Target Group — REST (port 6333) ──────────────────────────────────────────

resource "aws_lb_target_group" "qdrant_rest" {
  name     = "${var.name_prefix}-rest"
  port     = 6333
  protocol = "TCP"
  vpc_id   = data.aws_vpc.default.id

  health_check {
    protocol            = "HTTP"
    port                = "6333"
    path                = "/healthz"
    interval            = 10
    healthy_threshold   = 2
    unhealthy_threshold = 2
  }

  tags = {
    Name    = "${var.name_prefix}-tg-rest"
    Project = "card-oracle-max"
  }
}

# ── Target Group — gRPC (port 6334) ──────────────────────────────────────────

resource "aws_lb_target_group" "qdrant_grpc" {
  name     = "${var.name_prefix}-grpc"
  port     = 6334
  protocol = "TCP"
  vpc_id   = data.aws_vpc.default.id

  health_check {
    protocol            = "HTTP"
    port                = "6333"
    path                = "/healthz"
    interval            = 10
    healthy_threshold   = 2
    unhealthy_threshold = 2
  }

  tags = {
    Name    = "${var.name_prefix}-tg-grpc"
    Project = "card-oracle-max"
  }
}

# ── NLB Listeners ─────────────────────────────────────────────────────────────

resource "aws_lb_listener" "qdrant_rest" {
  load_balancer_arn = aws_lb.qdrant.arn
  port              = 6333
  protocol          = "TCP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.qdrant_rest.arn
  }
}

resource "aws_lb_listener" "qdrant_grpc" {
  load_balancer_arn = aws_lb.qdrant.arn
  port              = 6334
  protocol          = "TCP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.qdrant_grpc.arn
  }
}

# ── Target Group Attachments — seed + both peers ──────────────────────────────

resource "aws_lb_target_group_attachment" "rest_seed" {
  target_group_arn = aws_lb_target_group.qdrant_rest.arn
  target_id        = aws_instance.qdrant_seed.id
  port             = 6333
}

resource "aws_lb_target_group_attachment" "grpc_seed" {
  target_group_arn = aws_lb_target_group.qdrant_grpc.arn
  target_id        = aws_instance.qdrant_seed.id
  port             = 6334
}

resource "aws_lb_target_group_attachment" "rest_peer" {
  count            = 2
  target_group_arn = aws_lb_target_group.qdrant_rest.arn
  target_id        = aws_instance.qdrant_peer[count.index].id
  port             = 6333
}

resource "aws_lb_target_group_attachment" "grpc_peer" {
  count            = 2
  target_group_arn = aws_lb_target_group.qdrant_grpc.arn
  target_id        = aws_instance.qdrant_peer[count.index].id
  port             = 6334
}
