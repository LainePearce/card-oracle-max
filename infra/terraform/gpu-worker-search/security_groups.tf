# ── Lambda security group ─────────────────────────────────────────────────────
# Lambda in VPC — outbound to internal ALB only (port 80).
# No inbound rules needed (Lambda is never a server).
# HTTPS egress retained so Lambda can reach AWS APIs (S3, Secrets Manager, etc.)
# without a VPC endpoint.
#
# NOTE: The port-80 egress rule referencing internal_alb is defined as a
# standalone aws_security_group_rule below to avoid a Terraform dependency cycle
# (lambda_search ↔ internal_alb mutually reference each other).

resource "aws_security_group" "lambda_search" {
  name_prefix = "${var.name_prefix}-lambda-sg-"
  description = "Lambda search function - outbound to internal ALB only"
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "HTTPS to AWS services (S3, Secrets Manager, etc.)"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.name_prefix}-lambda-sg"
    Project = "card-oracle-max"
    Role    = "lambda-search"
  }

  lifecycle {
    create_before_destroy = true
    # Egress rules on this SG are managed by aws_security_group_rule resources.
    # Without this, Terraform fights the standalone rule and removes the port-80
    # egress to internal_alb on every plan/apply cycle.
    ignore_changes = [egress]
  }
}

# ── Internal ALB security group ───────────────────────────────────────────────
# Inbound: HTTP from Lambda SG only (see aws_security_group_rule below)
# Outbound: worker port to worker SG only (see aws_security_group_rule below)
#
# Both cross-SG rules are separated into aws_security_group_rule resources to
# avoid a cycle: lambda_search ↔ internal_alb.

resource "aws_security_group" "internal_alb" {
  name_prefix = "${var.name_prefix}-internal-alb-sg-"
  description = "Internal ALB for Lambda to GPU worker path - no public access"
  vpc_id      = data.aws_vpc.default.id

  tags = {
    Name    = "${var.name_prefix}-internal-alb-sg"
    Project = "card-oracle-max"
    Role    = "search-internal-alb"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Cross-SG rules added after both SGs exist (breaks the dependency cycle)
resource "aws_security_group_rule" "lambda_to_internal_alb" {
  description              = "Lambda HTTP egress to internal ALB"
  type                     = "egress"
  from_port                = 80
  to_port                  = 80
  protocol                 = "tcp"
  security_group_id        = aws_security_group.lambda_search.id
  source_security_group_id = aws_security_group.internal_alb.id
}

resource "aws_security_group_rule" "internal_alb_from_lambda" {
  description              = "Internal ALB HTTP ingress from Lambda only"
  type                     = "ingress"
  from_port                = 80
  to_port                  = 80
  protocol                 = "tcp"
  security_group_id        = aws_security_group.internal_alb.id
  source_security_group_id = aws_security_group.lambda_search.id
}

resource "aws_security_group_rule" "internal_alb_to_worker" {
  description              = "Internal ALB egress to GPU worker instances"
  type                     = "egress"
  from_port                = var.worker_port
  to_port                  = var.worker_port
  protocol                 = "tcp"
  security_group_id        = aws_security_group.internal_alb.id
  source_security_group_id = aws_security_group.search_worker.id
}

# ── ALB security group ────────────────────────────────────────────────────────
# Inbound: HTTP from allowed IPs only (my_ips for dev/operator access)
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
    description     = "Worker port from external ALB (operator/dev path)"
    from_port       = var.worker_port
    to_port         = var.worker_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  ingress {
    description     = "Worker port from internal ALB (Lambda path)"
    from_port       = var.worker_port
    to_port         = var.worker_port
    protocol        = "tcp"
    security_groups = [aws_security_group.internal_alb.id]
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
