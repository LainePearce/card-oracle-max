output "instance_id" {
  description = "EC2 instance ID of the daily worker"
  value       = aws_instance.daily_worker.id
}

output "public_ip" {
  description = "Public IP for SSH access"
  value       = aws_instance.daily_worker.public_ip
}

output "deploy_command" {
  description = "Rsync command to push code to the daily worker"
  value       = "rsync -av --exclude='.git' --exclude='__pycache__' --exclude='.venv' --exclude='*.pyc' ./ ec2-user@${aws_instance.daily_worker.public_ip}:~/card-oracle-max/"
}

output "ssh_command" {
  description = "SSH into the daily worker"
  value       = "ssh ec2-user@${aws_instance.daily_worker.public_ip}"
}

output "timer_status_command" {
  description = "Check daily update timer status on the worker"
  value       = "ssh ec2-user@${aws_instance.daily_worker.public_ip} 'sudo systemctl status daily-update.timer && sudo journalctl -u daily-update.service -n 50'"
}
