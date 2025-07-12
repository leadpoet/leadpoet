terraform {
  required_version = ">= 1.0"
  
  backend "s3" {
    bucket         = "leadpoet-terraform-state"
    key            = "leadpoet/terraform.tfstate"
    region         = "us-west-2"
    dynamodb_table = "leadpoet-terraform-locks"
    encrypt        = true
  }
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "${var.project}-${var.environment}"  cluster_version = "1.28"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    general = {
    general = {
      desired_capacity = var.eks_desired_capacity
      min_capacity     = var.eks_min_capacity
      max_capacity     = var.eks_max_capacity

      instance_types = var.eks_node_instance_types
      capacity_type  = "ON_DEMAND"
    }
  tags = var.common_tags
}

# VPC
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "leadpoet-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true

  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = var.common_tags
}

# RDS PostgreSQL with TimescaleDB
resource "aws_db_subnet_group" "leadpoet" {
  name       = "leadpoet-db-subnet-group"
  subnet_ids = module.vpc.private_subnets

  tags = var.common_tags
}

resource "aws_security_group" "rds" {
  name_prefix = "leadpoet-rds-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.common_tags
}

resource "aws_db_instance" "leadpoet" {
  identifier = "leadpoet-prod"

  engine         = "postgres"
  engine_version = "16.1"
  instance_class = "db.t3.medium"

  instance_class = var.db_instance_class  max_allocated_storage = 1000
  storage_type          = "gp3"
  storage_encrypted     = true

  db_name  = "leadpoet"
  username = "leadpoet_admin"
  password = random_password.db_password.result

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.leadpoet.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = false
  final_snapshot_identifier = "leadpoet-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"
  tags = var.common_tags
}

# Random password for database
resource "random_password" "db_password" {
  length  = 32
  special = true
}

# ElastiCache Redis Cluster
resource "aws_elasticache_subnet_group" "leadpoet" {
  name       = "leadpoet-redis-subnet-group"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name_prefix = "leadpoet-redis-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.common_tags
}

resource "aws_elasticache_replication_group" "leadpoet" {
  replication_group_id       = "leadpoet-redis"
  replication_group_description = "Leadpoet Redis cluster for caching"

  node_type                  = var.redis_node_type
  port                       = 6379
  parameter_group_name       = "default.redis7"
  automatic_failover_enabled = true
  num_cache_clusters         = 2

  subnet_group_name          = aws_elasticache_subnet_group.leadpoet.name
  security_group_ids         = [aws_security_group.redis.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  tags = var.common_tags
}

# MSK Kafka Cluster
resource "aws_msk_cluster" "leadpoet" {
  cluster_name           = "leadpoet-kafka"
  kafka_version          = "3.5.1"
  number_of_broker_nodes = 3

  broker_node_group_info {
    instance_type   = "kafka.t3.small"
    client_subnets  = module.vpc.private_subnets
    security_groups = [aws_security_group.kafka.id]
    storage_info {
    instance_type   = var.kafka_instance_type        volume_size = 100
      }
    }
  }

  encryption_info {
    encryption_in_transit {
      client_broker = "TLS"
      in_cluster    = true
    }
    encryption_at_rest_kms_key_arn = aws_kms_key.msk.arn
  }

  configuration_info {
    arn      = aws_msk_configuration.leadpoet.arn
    revision = aws_msk_configuration.leadpoet.latest_revision
  }

  tags = var.common_tags
}

resource "aws_security_group" "kafka" {
  name_prefix = "leadpoet-kafka-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 9092
    to_port         = 9092
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }

  ingress {
    from_port       = 9094
    to_port         = 9094
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.common_tags
}

resource "aws_kms_key" "msk" {
  description             = "KMS key for MSK encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = var.common_tags
}

resource "aws_msk_configuration" "leadpoet" {
  kafka_versions = ["3.5.1"]
  name           = "leadpoet-kafka-config"

  server_properties = <<PROPERTIES
auto.create.topics.enable=true
delete.topic.enable=true
log.retention.hours=168
log.retention.bytes=1073741824
PROPERTIES
}

# Outputs
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
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_replication_group.leadpoet.primary_endpoint_address
}

output "kafka_bootstrap_brokers" {
  description = "MSK Kafka bootstrap brokers"
  value       = aws_msk_cluster.leadpoet.bootstrap_brokers_tls
} 