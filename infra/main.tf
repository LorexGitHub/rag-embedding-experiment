terraform {
  required_providers {
    hcloud = {
      source  = "hetznercloud/hcloud"
      version = "~> 1.49"
    }
  }
}

variable "hcloud_token" {
  description = "Hetzner Cloud API token (set via TF_VAR_hcloud_token or prompt)"
  sensitive   = true
}

variable "ssh_public_key_path" {
  description = "Path to your public SSH key"
  default     = "~/.ssh/id_ed25519.pub"
}

provider "hcloud" {
  token = var.hcloud_token
}

resource "hcloud_ssh_key" "default" {
  name       = "rag-ensemble-key"
  public_key = file(var.ssh_public_key_path)
}

resource "hcloud_firewall" "rag" {
  name = "rag-ensemble-fw"

  rule {
    direction = "in"
    protocol  = "tcp"
    source_ips = ["0.0.0.0/0", "::/0"]
    port       = "22"
  }

  rule {
    direction = "in"
    protocol  = "tcp"
    source_ips = ["0.0.0.0/0", "::/0"]
    port       = "80"
  }

  rule {
    direction = "in"
    protocol  = "tcp"
    source_ips = ["0.0.0.0/0", "::/0"]
    port       = "443"
  }

  rule {
    direction = "in"
    protocol  = "tcp"
    source_ips = ["0.0.0.0/0", "::/0"]
    port       = "8000"
  }

  rule {
    direction = "in"
    protocol  = "tcp"
    source_ips = ["0.0.0.0/0", "::/0"]
    port       = "8501"
  }

  rule {
    direction = "in"
    protocol  = "tcp"
    source_ips = ["0.0.0.0/0", "::/0"]
    port       = "5100"
  }
}

resource "hcloud_server" "rag" {
  name        = "rag-ensemble"
  server_type = "cx23"
  image       = "ubuntu-24.04"
  location    = "nbg1"

  ssh_keys = [hcloud_ssh_key.default.id]
  firewall_ids = [hcloud_firewall.rag.id]

  user_data = <<-EOF
    #cloud-config
    package_update: true
    packages:
      - docker.io
      - docker-compose-v2
      - git

    runcmd:
      - systemctl enable --now docker
      - usermod -aG docker ubuntu
      - mkdir -p /opt/rag-ensemble
      - cd /opt/rag-ensemble
      - git clone https://github.com/LorexGitHub/rag-embedding-experiment.git .
      - docker compose pull
      - docker compose up -d
  EOF

  labels = {
    project = "rag-ensemble"
    tier    = "production"
  }
}

output "server_ip" {
  value = hcloud_server.rag.ipv4_address
}

output "server_ipv6" {
  value = hcloud_server.rag.ipv6_address
}

output "connect_cmd" {
  value = "ssh ubuntu@${hcloud_server.rag.ipv4_address}"
}
