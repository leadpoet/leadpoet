output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "cluster_iam_role_name" {
  description = "IAM role name associated with EKS cluster"
  value       = module.eks.cluster_iam_role_name
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "db_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = aws_db_instance.leadpoet.endpoint
  sensitive   = true
}

output "db_name" {
  description = "RDS PostgreSQL database name"
  value       = aws_db_instance.leadpoet.db_name
}

output "db_username" {
  description = "RDS PostgreSQL username"
  value       = aws_db_instance.leadpoet.username
  sensitive   = true
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_replication_group.leadpoet.primary_endpoint_address
}

output "redis_port" {
  description = "ElastiCache Redis port"
  value       = aws_elasticache_replication_group.leadpoet.port
}

output "kafka_bootstrap_brokers" {
  description = "MSK Kafka bootstrap brokers"
  value       = aws_msk_cluster.leadpoet.bootstrap_brokers_tls
  sensitive   = true
}

output "kafka_bootstrap_brokers_sasl_scram" {
  description = "MSK Kafka bootstrap brokers with SASL SCRAM"
  value       = aws_msk_cluster.leadpoet.bootstrap_brokers_sasl_scram
  sensitive   = true
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "private_subnets" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "Public subnet IDs"
  value       = module.vpc.public_subnets
} 