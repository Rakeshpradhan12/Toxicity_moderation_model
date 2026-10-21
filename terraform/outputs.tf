output "instance_id" {
  description = "ID of the created EC2 instance"
  value       = aws_instance.moderation_server.id
}

output "instance_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_instance.moderation_server.public_ip
}

output "instance_public_dns" {
  description = "Public DNS name of the EC2 instance"
  value       = aws_instance.moderation_server.public_dns
}

output "ssh_command" {
  description = "Command to SSH directly into the instance"
  value       = "ssh -i terraform/moderation-key.pem ubuntu@${aws_instance.moderation_server.public_ip}"
}

output "api_swagger_url" {
  description = "URL to access the interactive Swagger documentation"
  value       = "http://${aws_instance.moderation_server.public_ip}:8000/docs"
}

output "health_check_url" {
  description = "URL to check health status"
  value       = "http://${aws_instance.moderation_server.public_ip}:8000/health"
}

output "private_key_path" {
  description = "Local path to the downloaded SSH private key"
  value       = local_file.private_key.filename
}
