output "nlb_dns" {
  description = "NLB DNS name — use as OPENSEARCH_DOCS_HOST in .env"
  value       = aws_lb.opensearch.dns_name
}

output "master_private_ips" {
  description = "Private IPs of master nodes"
  value       = aws_instance.master[*].private_ip
}

output "data_private_ips" {
  description = "Private IPs of data nodes"
  value       = aws_instance.data[*].private_ip
}

output "cluster_security_group_id" {
  description = "Intra-cluster SG ID — reference from GPU worker and Lambda SG rules"
  value       = aws_security_group.opensearch_cluster.id
}

output "client_security_group_id" {
  description = "Client SG ID — attach to Lambda and GPU worker"
  value       = aws_security_group.opensearch_client.id
}

output "ssh_commands" {
  description = "SSH commands for each node"
  value = {
    masters = [for i, inst in aws_instance.master : "ssh -i ~/.ssh/${var.key_pair_name}.pem ec2-user@${inst.public_ip}"]
    data    = [for i, inst in aws_instance.data : "ssh -i ~/.ssh/${var.key_pair_name}.pem ec2-user@${inst.public_ip}"]
  }
}
