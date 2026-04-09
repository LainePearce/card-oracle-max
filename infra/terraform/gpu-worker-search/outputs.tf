output "alb_dns" {
  description = "ALB DNS name — set GPU_WORKER_URL=http://<this value> in your local .env"
  value       = aws_lb.search_worker.dns_name
}

output "alb_url" {
  description = "Full URL for the GPU worker API (via ALB)"
  value       = "http://${aws_lb.search_worker.dns_name}"
}

output "instance_public_ips" {
  description = "Public IPs of the GPU worker instances (for direct SSH access)"
  value       = [for i in aws_instance.search_worker : i.public_ip]
}

output "instance_ids" {
  description = "EC2 instance IDs"
  value       = [for i in aws_instance.search_worker : i.id]
}

output "health_check_url" {
  description = "Verify the worker is up: curl <this URL>"
  value       = "http://${aws_lb.search_worker.dns_name}/health"
}

output "deploy_command" {
  description = "Rsync command to push code to the first worker instance"
  value       = "rsync -av --exclude='.git' --exclude='__pycache__' --exclude='.venv' --exclude='*.pyc' ./ ec2-user@${aws_instance.search_worker[0].public_ip}:~/card-oracle-max/"
}

output "internal_alb_dns" {
  description = "Internal ALB DNS name — set GPU_WORKER_URL=http://<this value> in Lambda environment variables"
  value       = aws_lb.internal.dns_name
}

output "lambda_search_sg_id" {
  description = "Security group ID to attach to the Lambda function for GPU worker access"
  value       = aws_security_group.lambda_search.id
}
