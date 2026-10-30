variable "tenancy_ocid" {
  description = "OCID of the tenancy (Oracle console: Tenancy details)"
  type        = string
}

variable "user_ocid" {
  description = "OCID of the API user whose key signs Terraform requests"
  type        = string
}

variable "fingerprint" {
  description = "Fingerprint of the API signing key, from the console"
  type        = string
}

variable "private_key_path" {
  description = "Path to the API signing private key (PEM) on this machine"
  type        = string
  default     = "~/.oci/oci_api_key.pem"
}

variable "region" {
  description = "OCI region identifier, e.g. ap-mumbai-1"
  type        = string
}

variable "compartment_ocid" {
  description = "Compartment to create resources in; the tenancy OCID works"
  type        = string
}

variable "shape" {
  description = "Compute shape. VM.Standard.A1.Flex is the free ARM tier"
  type        = string
  default     = "VM.Standard.A1.Flex"
}

variable "ocpus" {
  description = "OCPUs for the A1 shape (Always Free allows up to 4)"
  type        = number
  default     = 2
}

variable "memory_in_gbs" {
  description = "RAM for the A1 shape (Always Free allows up to 24)"
  type        = number
  default     = 12
}

variable "ssh_cidr" {
  description = "CIDR allowed to SSH, e.g. 203.0.113.4/32. Never 0.0.0.0/0"
  type        = string
}

variable "ssh_public_key_path" {
  description = "Path to the public key installed for the ubuntu user"
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}

variable "image" {
  description = "Container image the VM runs, published by CI"
  type        = string
}

variable "domain" {
  description = "Public hostname for TLS. Empty uses <ip>.sslip.io, which is a real certificate for a free hostname"
  type        = string
  default     = ""
}

variable "clerk_issuer" {
  description = "Clerk issuer URL. Empty runs the deployment without sign-in"
  type        = string
  default     = ""
}

variable "clerk_publishable_key" {
  description = "Clerk publishable key (safe to expose; the secret key is never needed by this stack)"
  type        = string
  default     = ""
  sensitive   = false
}

variable "github_actor" {
  description = "GitHub username, for pulling a private GHCR image"
  type        = string
  default     = ""
}

variable "github_token" {
  description = "GitHub token with read:packages, for a private GHCR image. Empty for a public package"
  type        = string
  default     = ""
  sensitive   = true
}
