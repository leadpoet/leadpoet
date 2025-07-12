#!/bin/bash

# Leadpoet Terraform Backend Setup Script
# This script sets up the S3 bucket and DynamoDB table for Terraform state management

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUCKET_NAME="leadpoet-terraform-state"
DYNAMODB_TABLE="leadpoet-terraform-locks"
REGION="us-west-2"

echo -e "${BLUE}🚀 Leadpoet Terraform Backend Setup${NC}"
echo "This script will create the S3 bucket and DynamoDB table for Terraform state management."
echo ""

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo -e "${RED}❌ AWS CLI is not configured. Please run 'aws configure' first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ AWS CLI is configured${NC}"

# Check if bucket already exists
if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  S3 bucket '$BUCKET_NAME' already exists${NC}"
else
    echo -e "${BLUE}Creating S3 bucket '$BUCKET_NAME'...${NC}"
    aws s3api create-bucket \
        --bucket "$BUCKET_NAME" \
        --region "$REGION" \
        --create-bucket-configuration LocationConstraint="$REGION"
    
    # Enable versioning
    aws s3api put-bucket-versioning \
        --bucket "$BUCKET_NAME" \
        --versioning-configuration Status=Enabled
    
    # Enable encryption
    aws s3api put-bucket-encryption \
        --bucket "$BUCKET_NAME" \
        --server-side-encryption-configuration '{
            "Rules": [
                {
                    "ApplyServerSideEncryptionByDefault": {
                        "SSEAlgorithm": "AES256"
                    }
                }
            ]
        }'
    
    # Block public access
    aws s3api put-public-access-block \
        --bucket "$BUCKET_NAME" \
        --public-access-block-configuration \
        BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
    
    echo -e "${GREEN}✓ S3 bucket created and configured${NC}"
fi

# Check if DynamoDB table already exists
if aws dynamodb describe-table --table-name "$DYNAMODB_TABLE" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  DynamoDB table '$DYNAMODB_TABLE' already exists${NC}"
else
    echo -e "${BLUE}Creating DynamoDB table '$DYNAMODB_TABLE'...${NC}"
    aws dynamodb create-table \
        --table-name "$DYNAMODB_TABLE" \
        --attribute-definitions AttributeName=LockID,AttributeType=S \
        --key-schema AttributeName=LockID,KeyType=HASH \
        --billing-mode PAY_PER_REQUEST \
        --region "$REGION"
    
    # Wait for table to be active
    echo "Waiting for DynamoDB table to be active..."
    aws dynamodb wait table-exists --table-name "$DYNAMODB_TABLE"
    
    echo -e "${GREEN}✓ DynamoDB table created${NC}"
fi

echo ""
echo -e "${GREEN}✅ Backend infrastructure setup complete!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Initialize Terraform with the backend:"
echo "   terraform init"
echo ""
echo "2. Plan your infrastructure:"
echo "   terraform plan"
echo ""
echo "3. Apply your infrastructure:"
echo "   terraform apply"
echo ""
echo -e "${YELLOW}Note: The backend configuration is already set in main.tf${NC}"
echo "Backend details:"
echo "- S3 Bucket: $BUCKET_NAME"
echo "- DynamoDB Table: $DYNAMODB_TABLE"
echo "- Region: $REGION"
echo "- State Key: leadpoet/terraform.tfstate" 