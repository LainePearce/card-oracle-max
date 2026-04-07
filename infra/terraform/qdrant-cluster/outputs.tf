output "nlb_dns_name" {
  description = "NLB DNS name — use as QDRANT_HOST in .env"
  value       = aws_lb.qdrant.dns_name
}

output "qdrant_rest_url" {
  description = "Qdrant REST API via NLB"
  value       = "http://${aws_lb.qdrant.dns_name}:6333"
}

output "qdrant_grpc_endpoint" {
  description = "Qdrant gRPC endpoint via NLB (host:port)"
  value       = "${aws_lb.qdrant.dns_name}:6334"
}

output "node_private_ips" {
  description = "Private IPs of all cluster nodes [seed, peer-0, peer-1]"
  value = [
    aws_instance.qdrant_seed.private_ip,
    aws_instance.qdrant_peer[0].private_ip,
    aws_instance.qdrant_peer[1].private_ip,
  ]
}

output "node_public_ips" {
  description = "Public IPs of all cluster nodes (for direct SSH access)"
  value = [
    aws_instance.qdrant_seed.public_ip,
    aws_instance.qdrant_peer[0].public_ip,
    aws_instance.qdrant_peer[1].public_ip,
  ]
}

output "ssh_commands" {
  description = "SSH commands for each node"
  value = [
    "ssh -i ~/.ssh/${var.key_pair_name}.pem ec2-user@${aws_instance.qdrant_seed.public_ip}     # node-0 (seed)",
    "ssh -i ~/.ssh/${var.key_pair_name}.pem ec2-user@${aws_instance.qdrant_peer[0].public_ip}  # node-1",
    "ssh -i ~/.ssh/${var.key_pair_name}.pem ec2-user@${aws_instance.qdrant_peer[1].public_ip}  # node-2",
  ]
}

output "cluster_health_url" {
  description = "Cluster health check URL"
  value       = "http://${aws_lb.qdrant.dns_name}:6333/cluster"
}

output "env_snippet" {
  description = "Paste these into .env after apply"
  value       = <<-ENV
    QDRANT_HOST=${aws_lb.qdrant.dns_name}
    QDRANT_PORT=6334
    QDRANT_USE_GRPC=true
    QDRANT_COLLECTION=cards
  ENV
}
