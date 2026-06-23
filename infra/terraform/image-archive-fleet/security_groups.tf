resource "aws_security_group" "worker" {
  name_prefix = "${var.name_prefix}-sg-"
  description = "Image-archive CPU worker - SSH + dashboard from operator, all outbound"

  ingress {
    description = "SSH from operator"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.my_ips
  }

  ingress {
    description = "Image-archive dashboard (8082) from operator"
    from_port   = 8082
    to_port     = 8082
    protocol    = "tcp"
    cidr_blocks = var.my_ips
  }

  ingress {
    description = "Intra-fleet SSH (dashboard polls workers)"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    self        = true
  }

  # Outbound: OpenSearch, S3, image downloads (eBay CDN), pip
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.name_prefix}-sg"
    Project = "card-oracle-max"
  }

  lifecycle {
    create_before_destroy = true
  }
}
