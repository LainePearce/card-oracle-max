output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.qdrant.id
}

output "public_ip" {
  description = "Public IP of the Qdrant test instance"
  value       = var.assign_eip ? aws_eip.qdrant[0].public_ip : aws_instance.qdrant.public_ip
}

output "qdrant_http_url" {
  description = "Qdrant HTTP REST API endpoint"
  value       = "http://${var.assign_eip ? aws_eip.qdrant[0].public_ip : aws_instance.qdrant.public_ip}:6333"
}

output "qdrant_grpc_endpoint" {
  description = "Qdrant gRPC endpoint (host:port)"
  value       = "${var.assign_eip ? aws_eip.qdrant[0].public_ip : aws_instance.qdrant.public_ip}:6334"
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ~/.ssh/${var.key_pair_name}.pem ec2-user@${var.assign_eip ? aws_eip.qdrant[0].public_ip : aws_instance.qdrant.public_ip}"
}
