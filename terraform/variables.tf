variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "ap-south-1"
}

variable "instance_type" {
  description = "EC2 instance type (c7i-flex.large is 2 vCPU, 4 GiB RAM, Free tier eligible)"
  type        = string
  default     = "c7i-flex.large"
}

variable "volume_size_gb" {
  description = "Root EBS storage size in GB (Free tier covers up to 30 GB gp3)"
  type        = number
  default     = 30
}

variable "instance_name" {
  description = "Name tag for the EC2 instance"
  type        = string
  default     = "toxicity-moderation-server"
}

variable "allowed_ssh_cidr" {
  description = "CIDR block permitted to SSH into the instance"
  type        = string
  default     = "0.0.0.0/0"
}
