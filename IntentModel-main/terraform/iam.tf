# IAM Roles and Policies for Leadpoet Intent Model
# Following least-privilege principle

# EKS Cluster IAM Role
resource "aws_iam_role" "eks_cluster_role" {
  name = "leadpoet-eks-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })

  tags = var.common_tags
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster_role.name
}

# EKS Node Group IAM Role
resource "aws_iam_role" "eks_node_group_role" {
  name = "leadpoet-eks-node-group-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = var.common_tags
}

resource "aws_iam_role_policy_attachment" "eks_worker_node_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_node_group_role.name
}

resource "aws_iam_role_policy_attachment" "eks_cni_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_node_group_role.name
}

resource "aws_iam_role_policy_attachment" "ec2_container_registry_read_only" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_node_group_role.name
}

# S3 Bucket for Application Data
resource "aws_s3_bucket" "leadpoet_data" {
  bucket = "leadpoet-intent-model-data-${random_string.bucket_suffix.result}"

  tags = merge(var.common_tags, {
    Name = "leadpoet-intent-model-data"
  })
}

# S3 Bucket for Terraform State (if not using existing)
resource "aws_s3_bucket" "terraform_state" {
  bucket = "leadpoet-terraform-state-${random_string.bucket_suffix.result}"

  tags = merge(var.common_tags, {
    Name = "leadpoet-terraform-state"
  })
}

# S3 Bucket Versioning
resource "aws_s3_bucket_versioning" "leadpoet_data_versioning" {
  bucket = aws_s3_bucket.leadpoet_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "terraform_state_versioning" {
  bucket = aws_s3_bucket.terraform_state.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 Bucket Encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "leadpoet_data_encryption" {
  bucket = aws_s3_bucket.leadpoet_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state_encryption" {
  bucket = aws_s3_bucket.terraform_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

# S3 Bucket Public Access Block
resource "aws_s3_bucket_public_access_block" "leadpoet_data_public_access_block" {
  bucket = aws_s3_bucket.leadpoet_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "terraform_state_public_access_block" {
  bucket = aws_s3_bucket.terraform_state.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# IAM Policy for Application S3 Access (Least Privilege)
resource "aws_iam_policy" "leadpoet_s3_access" {
  name        = "leadpoet-s3-access-policy"
  description = "Least-privilege S3 access policy for Leadpoet Intent Model"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion"
        ]
        Resource = [
          "${aws_s3_bucket.leadpoet_data.arn}/models/*",
          "${aws_s3_bucket.leadpoet_data.arn}/config/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.leadpoet_data.arn}/logs/*",
          "${aws_s3_bucket.leadpoet_data.arn}/exports/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.leadpoet_data.arn
        ]
        Condition = {
          StringEquals = {
            "s3:prefix" = [
              "models/",
              "config/",
              "logs/",
              "exports/"
            ]
          }
        }
      }
    ]
  })
}

# IAM Policy for Terraform State Access
resource "aws_iam_policy" "terraform_state_access" {
  name        = "leadpoet-terraform-state-access"
  description = "Access policy for Terraform state bucket"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.terraform_state.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.terraform_state.arn
        ]
      }
    ]
  })
}

# IAM Policy for DynamoDB State Locking
resource "aws_iam_policy" "dynamodb_state_locking" {
  name        = "leadpoet-dynamodb-state-locking"
  description = "Access policy for DynamoDB state locking"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:DeleteItem"
        ]
        Resource = [
          aws_dynamodb_table.terraform_locks.arn
        ]
      }
    ]
  })
}

# Service Account IAM Role for EKS
resource "aws_iam_role" "leadpoet_service_account_role" {
  name = "leadpoet-service-account-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/${replace(aws_eks_cluster.leadpoet.identity[0].oidc[0].issuer, "https://", "")}"
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${replace(aws_eks_cluster.leadpoet.identity[0].oidc[0].issuer, "https://", "")}:sub" = "system:serviceaccount:leadpoet-prod:leadpoet-api"
          }
        }
      }
    ]
  })

  tags = var.common_tags
}

# Attach S3 access policy to service account role
resource "aws_iam_role_policy_attachment" "leadpoet_s3_access_attachment" {
  role       = aws_iam_role.leadpoet_service_account_role.name
  policy_arn = aws_iam_policy.leadpoet_s3_access.arn
}

# IAM Policy for Secrets Manager Access
resource "aws_iam_policy" "secrets_manager_access" {
  name        = "leadpoet-secrets-manager-access"
  description = "Access policy for AWS Secrets Manager"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = [
          "arn:aws:secretsmanager:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:secret:leadpoet-*"
        ]
      }
    ]
  })
}

# Attach Secrets Manager policy to service account role
resource "aws_iam_role_policy_attachment" "leadpoet_secrets_access_attachment" {
  role       = aws_iam_role.leadpoet_service_account_role.name
  policy_arn = aws_iam_policy.secrets_manager_access.arn
}

# Random string for bucket names
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {} 