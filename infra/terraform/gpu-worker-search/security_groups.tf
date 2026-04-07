# ── ALB security group ────────────────────────────────────────────────────────
# Inbound: HTTP from allowed IPs only (my_ips for initial dev testing;
#          open to 0.0.0.0/0 when ready for production traffic)
# Outbound: HTTP to GPU worker port (restricted to worker SG)

resource "aws_security_group" "alb" {
  name_prefix = "${var.name_prefix}-alb-sg-"
  description = "GPU search worker ALB - HTTP inbound from allowed IPs"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "HTTP from allowed client IPs"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.my_ips
  }

  egress {
    description = "Forward to GPU worker instances"
    from_port   = var.worker_port
    to_port     = var.worker_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.name_prefix}-alb-sg"
    Project = "card-oracle-max"
    Role    = "search-alb"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# ── GPU worker security group ─────────────────────────────────────────────────
# Inbound: worker port from ALB SG only; SSH from operator IPs
# Outbound: all (Qdrant, OpenSearch, CDN image downloads, pip installs)

resource "aws_security_group" "search_worker" {
  name_prefix = "${var.name_prefix}-sg-"
  description = "GPU search worker instance - accepts traffic from ALB and SSH from operators"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description     = "Worker port from ALB only"
    from_port       = var.worker_port
    to_port         = var.worker_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  ingress {
    description = "SSH from operator IPs"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.my_ips
  }

  egress {
    description = "All outbound (Qdrant, OpenSearch, CDN, package installs)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.name_prefix}-sg"
    Project = "card-oracle-max"
    Role    = "search-worker"
  }

  lifecycle {
    create_before_destroy = true
  }
}
