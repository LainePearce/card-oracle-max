output "asg_name" {
  description = "Auto Scaling Group name"
  value       = aws_autoscaling_group.worker.name
}

output "launch_template_id" {
  value = aws_launch_template.worker.id
}

output "list_instances_cmd" {
  description = "Show current (dynamic) fleet IPs — ASG instances are replaced automatically, so IPs change."
  value       = "aws ec2 describe-instances --filters 'Name=tag:Name,Values=${var.name_prefix}-worker' 'Name=instance-state-name,Values=running' --query 'Reservations[].Instances[].PublicIpAddress' --output text"
}

output "refresh_cmd" {
  description = "Roll the fleet onto new code/AMI after publishing a new tarball"
  value       = "aws autoscaling start-instance-refresh --auto-scaling-group-name ${aws_autoscaling_group.worker.name}"
}
