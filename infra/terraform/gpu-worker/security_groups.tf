resource "aws_security_group" "gpu_worker" {
  name_prefix = "${var.name_prefix}-sg-"
  description = "GPU worker - SSH + dashboard from operator IP, all outbound"

  # SSH
  ingress {
    description = "SSH from operator"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.my_ips
  }

  # Backfill dashboard (worker-0 only, but rule applied fleet-wide for simplicity)
  ingress {
    description = "Backfill dashboard from operator"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = var.my_ips
  }

  # Intra-fleet SSH: workers can reach each other for dashboard SSH polling
  ingress {
    description = "Intra-fleet SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    self        = true
  }

  # All outbound (needed for: OpenSearch, S3, Qdrant, image downloads, pip)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.name_prefix}-sg"
    Project = "card-oracle-max"
    Stage   = "1"
  }

  lifecycle {
    create_before_destroy = true
  }
}
