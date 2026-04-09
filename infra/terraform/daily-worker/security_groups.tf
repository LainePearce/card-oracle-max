# ── Daily worker security group ───────────────────────────────────────────────
# SSH from operator IPs only.
# All outbound for Qdrant, RDS, S3, CDN image downloads.
# No inbound HTTP — this worker processes batch jobs, not web requests.

resource "aws_security_group" "daily_worker" {
  name_prefix = "${var.name_prefix}-sg-"
  description = "Daily embedding update worker - SSH from operators, all outbound"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH from operator IPs"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.my_ips
  }

  egress {
    description = "All outbound (Qdrant, RDS, S3, CDN image downloads)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.name_prefix}-sg"
    Project = "card-oracle-max"
    Role    = "daily-worker"
  }

  lifecycle {
    create_before_destroy = true
  }
}
