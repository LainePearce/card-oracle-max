output "public_ips" {
  description = "Public IPs of the CPU image-archive workers"
  value       = aws_instance.worker[*].public_ip
}

output "private_ips" {
  description = "Private IPs (intra-fleet SSH / dashboard polling)"
  value       = aws_instance.worker[*].private_ip
}

output "instance_ids" {
  value = aws_instance.worker[*].id
}

output "deploy_hint" {
  description = "How to push code to the fleet"
  value       = "infra/scripts/deploy-image-archive-fleet.sh ${join(" ", aws_instance.worker[*].public_ip)}"
}
