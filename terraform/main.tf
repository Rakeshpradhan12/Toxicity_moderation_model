terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.5"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Lookup default VPC in the region
data "aws_vpc" "default" {
  default = true
}

# ----------------------------------------------------------------------------
# 1. Automatic SSH Key Pair Generation
# ----------------------------------------------------------------------------
resource "tls_private_key" "ssh_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "generated_key" {
  key_name   = "${var.instance_name}-key"
  public_key = tls_private_key.ssh_key.public_key_openssh

  tags = {
    Name = "${var.instance_name}-key"
  }
}

# Save the private key locally as moderation-key.pem
resource "local_file" "private_key" {
  content         = tls_private_key.ssh_key.private_key_pem
  filename        = "${path.module}/moderation-key.pem"
  file_permission = "0600"
}

# ----------------------------------------------------------------------------
# 2. Security Group (Firewall)
# ----------------------------------------------------------------------------
resource "aws_security_group" "moderation_sg" {
  name        = "${var.instance_name}-sg"
  description = "Security group for Toxicity & Content Moderation Server"
  vpc_id      = data.aws_vpc.default.id

  # SSH access
  ingress {
    description = "SSH access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  # FastAPI API & Swagger UI
  ingress {
    description = "FastAPI Moderation Service"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Standard HTTP
  ingress {
    description = "HTTP access"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Outbound access (for Docker installs, package updates, and S3 downloads)
  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.instance_name}-sg"
  }
}

# ----------------------------------------------------------------------------
# 3. AMI Lookup (Ubuntu 24.04 LTS x86_64)
# ----------------------------------------------------------------------------
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ----------------------------------------------------------------------------
# 4. EC2 Instance Provisioning
# ----------------------------------------------------------------------------
resource "aws_instance" "moderation_server" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = var.instance_type
  key_name                    = aws_key_pair.generated_key.key_name
  vpc_security_group_ids      = [aws_security_group.moderation_sg.id]
  associate_public_ip_address = true

  root_block_device {
    volume_size           = var.volume_size_gb
    volume_type           = "gp3"
    delete_on_termination = true

    tags = {
      Name = "${var.instance_name}-root-volume"
    }
  }

  # Automated server bootstrap script
  user_data = <<-EOF
              #!/bin/bash
              set -e

              # 1. Setup 4 GB swap file for safety
              if [ ! -f /swapfile ]; then
                  fallocate -l 4G /swapfile
                  chmod 600 /swapfile
                  mkswap /swapfile
                  swapon /swapfile
                  echo '/swapfile none swap sw 0 0' >> /etc/fstab
              fi

              # 2. Update and install packages
              apt-get update -y
              apt-get install -y ca-certificates curl gnupg git

              # 3. Install Docker and Docker Compose plugin
              install -m 0755 -d /etc/apt/keyrings
              curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
              chmod a+r /etc/apt/keyrings/docker.asc
              echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
              apt-get update -y
              apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

              # 4. Enable and start Docker service
              systemctl enable docker
              systemctl start docker
              usermod -aG docker ubuntu

              # 5. Clone production branch from GitHub
              mkdir -p /home/ubuntu/app
              git clone -b production https://github.com/Rakeshpradhan12/Toxicity_moderation_model.git /home/ubuntu/app

              # 6. Securely write .env from local machine (never committed to GitHub!)
              cat <<'ENV_EOF' > /home/ubuntu/app/.env
${file("${path.module}/../.env")}
ENV_EOF
              chown -R ubuntu:ubuntu /home/ubuntu/app

              # 7. Launch application with Docker Compose
              cd /home/ubuntu/app
              docker compose up -d --build
              EOF

  tags = {
    Name = var.instance_name
  }
}
