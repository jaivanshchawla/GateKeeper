# Oracle Cloud Always Free: one ARM VM that runs the whole product.
#
# This is the long-term deployment, because it is the only free tier that gives
# a real always-on machine with enough RAM to run the API, the dashboard,
# Prometheus and Grafana on one box -- no cold starts, nothing that sleeps.
#
# Everything stays inside Oracle's Always Free limits:
#   VM.Standard.A1.Flex, 2 OCPU / 12 GB   (of 4 / 24 available)
#   one 50 GB boot volume                 (of 200 GB available)
#   one public IPv4                       (of 4 available)
#
# Usage:
#   cp terraform.tfvars.example terraform.tfvars   # fill in OCI credentials
#   terraform init && terraform apply
#
# The instance boots with cloud-init, which installs Docker and starts the stack
# from the image published by .github/workflows/deploy.yml.

terraform {
  required_version = ">= 1.5"
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = "~> 6.0"
    }
  }
}

provider "oci" {
  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid
  fingerprint      = var.fingerprint
  private_key_path = var.private_key_path
  region           = var.region
}

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

# Ubuntu 22.04 for the instance's shape: arm64 for A1, amd64 for E2.
data "oci_core_images" "ubuntu" {
  compartment_id           = var.compartment_ocid
  operating_system         = "Canonical Ubuntu"
  operating_system_version = "22.04"
  shape                    = var.shape
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

resource "oci_core_vcn" "gatekeeper" {
  compartment_id = var.compartment_ocid
  cidr_blocks    = ["10.0.0.0/16"]
  display_name   = "gatekeeper-vcn"
  dns_label      = "gatekeeper"
}

resource "oci_core_internet_gateway" "gatekeeper" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.gatekeeper.id
  display_name   = "gatekeeper-igw"
  enabled        = true
}

resource "oci_core_route_table" "gatekeeper" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.gatekeeper.id
  display_name   = "gatekeeper-rt"

  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_internet_gateway.gatekeeper.id
  }
}

resource "oci_core_security_list" "gatekeeper" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.gatekeeper.id
  display_name   = "gatekeeper-sl"

  # SSH is limited to the operator's address, never the whole internet.
  ingress_security_rules {
    description = "ssh from operator"
    protocol    = "6" # TCP
    source      = var.ssh_cidr
    tcp_options {
      min = 22
      max = 22
    }
  }

  ingress_security_rules {
    description = "http, for ACME challenges and the redirect"
    protocol    = "6"
    source      = "0.0.0.0/0"
    tcp_options {
      min = 80
      max = 80
    }
  }

  ingress_security_rules {
    description = "https"
    protocol    = "6"
    source      = "0.0.0.0/0"
    tcp_options {
      min = 443
      max = 443
    }
  }

  egress_security_rules {
    description      = "all outbound"
    protocol         = "all"
    destination      = "0.0.0.0/0"
    destination_type = "CIDR_BLOCK"
  }
}

resource "oci_core_subnet" "gatekeeper" {
  compartment_id             = var.compartment_ocid
  vcn_id                     = oci_core_vcn.gatekeeper.id
  cidr_block                 = "10.0.1.0/24"
  display_name               = "gatekeeper-subnet"
  dns_label                  = "app"
  route_table_id             = oci_core_route_table.gatekeeper.id
  security_list_ids          = [oci_core_security_list.gatekeeper.id]
  prohibit_public_ip_on_vnic = false
}

resource "oci_core_instance" "gatekeeper" {
  compartment_id      = var.compartment_ocid
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  display_name        = "gatekeeper"
  shape               = var.shape

  dynamic "shape_config" {
    for_each = startswith(var.shape, "VM.Standard.A1") ? [1] : []
    content {
      ocpus         = var.ocpus
      memory_in_gbs = var.memory_in_gbs
    }
  }

  source_details {
    source_type = "image"
    source_id   = data.oci_core_images.ubuntu.images[0].id
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.gatekeeper.id
    assign_public_ip = true
    display_name     = "gatekeeper-vnic"
    hostname_label   = "gatekeeper"
  }

  metadata = {
    ssh_authorized_keys = trimspace(file(var.ssh_public_key_path))
    user_data = base64encode(templatefile("${path.module}/cloud-init.yaml.tftpl", {
      image                 = var.image
      domain                = var.domain
      clerk_issuer          = var.clerk_issuer
      clerk_publishable_key = var.clerk_publishable_key
      github_actor          = var.github_actor
      github_token          = var.github_token
    }))
  }

  lifecycle {
    # user_data is read on first boot only. Replacing the instance would also
    # replace its boot volume, and with it the SQLite database.
    ignore_changes = [metadata]
  }
}
