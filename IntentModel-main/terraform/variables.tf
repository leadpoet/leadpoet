variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}
variable "project" {
  description = "Project name"
  type        = string
  default     = "leadpoet"
}

variable "common_tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
variable "common_tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project     = "leadpoet"
    ManagedBy   = "terraform"
    Owner       = "engineering"
  }
}
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "kafka_instance_type" {
  description = "MSK Kafka broker instance type"
  type        = string
  default     = "kafka.t3.small"
}

variable "eks_node_instance_types" {
  description = "EKS node instance types"
  type        = list(string)
  default     = ["t3.medium"]
}

variable "eks_desired_capacity" {
  description = "Desired number of EKS nodes"
  type        = number
  default     = 3
}

variable "eks_min_capacity" {
  description = "Minimum number of EKS nodes"
  type        = number
  default     = 3
}

variable "eks_max_capacity" {
  description = "Maximum number of EKS nodes"
  type        = number
  default     = 10
} 