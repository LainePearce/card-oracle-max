# ── Intra-cluster SG (transport + HTTP between nodes) ────────────────────────

resource "aws_security_group" "opensearch_cluster" {
  name_prefix = "${var.cluster_name}-cluster-"
  description = "OpenSearch inter-node traffic (transport 9300, HTTP 9200)"
  vpc_id      = var.vpc_id

  # Transport layer - inter-node clustering
  ingress {
    description = "OpenSearch transport between nodes"
    from_port   = 9300
    to_port     = 9400
    protocol    = "tcp"
    self        = true
  }

  # HTTP - inter-node (coordinating queries)
  ingress {
    description = "OpenSearch HTTP between nodes"
    from_port   = 9200
    to_port     = 9200
    protocol    = "tcp"
    self        = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.cluster_name}-cluster-sg"
    Project = "card-oracle-max"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# ── Client access SG (Lambda, GPU workers, operators) ────────────────────────

resource "aws_security_group" "opensearch_client" {
  name_prefix = "${var.cluster_name}-client-"
  description = "OpenSearch client access on port 9200"
  vpc_id      = var.vpc_id

  # Operator access (SSH for admin tasks)
  ingress {
    description = "SSH from operator"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.operator_ips
  }

  # HTTP from operators (curl, Kibana/Dashboards, backfill workers)
  ingress {
    description = "OpenSearch HTTP from operators"
    from_port   = 9200
    to_port     = 9200
    protocol    = "tcp"
    cidr_blocks = var.operator_ips
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.cluster_name}-client-sg"
    Project = "card-oracle-max"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# ── Allow specific SGs (Lambda, GPU worker) to reach port 9200 ───────────────

resource "aws_security_group_rule" "allow_client_sg" {
  count = length(var.client_security_group_ids)

  type                     = "ingress"
  from_port                = 9200
  to_port                  = 9200
  protocol                 = "tcp"
  security_group_id        = aws_security_group.opensearch_client.id
  source_security_group_id = var.client_security_group_ids[count.index]
  description              = "OpenSearch HTTP from client SG ${count.index}"
}
