data "aws_vpc" "default" {
  default = true
}

resource "aws_security_group" "qdrant_cluster" {
  name        = "${var.name_prefix}-sg"
  description = "Qdrant cluster nodes: REST, gRPC, P2P, SSH"
  vpc_id      = data.aws_vpc.default.id

  # SSH — management access
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  # REST API — from your machine + anything in the VPC (GPU worker, repopulate scripts)
  ingress {
    description = "Qdrant REST"
    from_port   = 6333
    to_port     = 6333
    protocol    = "tcp"
    cidr_blocks = [var.my_ip, data.aws_vpc.default.cidr_block]
  }

  # gRPC — from your machine + VPC (GPU worker)
  ingress {
    description = "Qdrant gRPC"
    from_port   = 6334
    to_port     = 6334
    protocol    = "tcp"
    cidr_blocks = [var.my_ip, data.aws_vpc.default.cidr_block]
  }

  # P2P cluster gossip — only between cluster nodes (self-referencing)
  ingress {
    description = "Qdrant cluster P2P"
    from_port   = 6335
    to_port     = 6335
    protocol    = "tcp"
    self        = true
  }

  # Full egress — needed for Docker image pull, S3 repopulation, NTP
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
}
