output "worker_public_ips" {
  description = "Public IPs of all GPU backfill workers"
  value       = aws_instance.gpu_worker[*].public_ip
}

output "ssh_commands" {
  description = "SSH commands to connect to each worker"
  value = [
    for i, inst in aws_instance.gpu_worker :
    "ssh -i ~/.ssh/${var.key_pair_name}.pem ec2-user@${inst.public_ip}  # worker-${i}"
  ]
}

output "backfill_start_commands" {
  description = "Commands to start the backfill service on each worker (run after deploying code)"
  value = [
    for i, inst in aws_instance.gpu_worker :
    "ssh -i ~/.ssh/${var.key_pair_name}.pem ec2-user@${inst.public_ip} 'sudo systemctl start backfill && sudo journalctl -fu backfill'  # worker-${i}"
  ]
}

output "worker_instance_ids" {
  description = "EC2 instance IDs"
  value       = aws_instance.gpu_worker[*].id
}
