resource "aws_security_group" "gpu_worker" {
  name_prefix = "${var.name_prefix}-sg-"
  description = "GPU worker - SSH from operator IP, all outbound"

  # SSH
  ingress {
    description = "SSH from operator"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.my_ips
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
