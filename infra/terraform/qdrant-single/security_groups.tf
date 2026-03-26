resource "aws_security_group" "qdrant" {
  name_prefix = "${var.name_prefix}-sg-"
  description = "Qdrant test instance - HTTP, gRPC, SSH from operator IP"

  # SSH
  ingress {
    description = "SSH from operator"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  # Qdrant HTTP REST API
  ingress {
    description = "Qdrant HTTP REST"
    from_port   = 6333
    to_port     = 6333
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  # Qdrant gRPC API
  ingress {
    description = "Qdrant gRPC"
    from_port   = 6334
    to_port     = 6334
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  # All outbound
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
