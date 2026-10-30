output "public_ip" {
  description = "The VM's public address"
  value       = oci_core_instance.gatekeeper.public_ip
}

output "url" {
  description = "Where the dashboard is served once cloud-init finishes (a few minutes)"
  value       = var.domain != "" ? "https://${var.domain}" : "https://${replace(oci_core_instance.gatekeeper.public_ip, ".", "-")}.sslip.io"
}

output "ssh" {
  description = "SSH command for the instance"
  value       = "ssh ubuntu@${oci_core_instance.gatekeeper.public_ip}"
}

output "reserve_the_ip" {
  description = "Oracle reassigns ephemeral IPs on stop/start. To keep a permanent address, reserve the instance's public IP in the console (Networking -> IP management)."
  value       = "https://cloud.oracle.com/networking/ip-management?region=${var.region}"
}
