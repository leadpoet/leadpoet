#!/bin/bash

# Leadpoet API Deployment Script
# Deploys the Leadpoet Intent Model API to Kubernetes using Helm

set -euo pipefail

# Configuration
CHART_NAME="leadpoet-api"
CHART_PATH="./helm/leadpoet-api"
NAMESPACE="leadpoet-prod"
RELEASE_NAME="leadpoet-api"
VALUES_FILE=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --values)
            VALUES_FILE="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --release-name)
            RELEASE_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --values FILE       Path to values file"
            echo "  --namespace NAME    Kubernetes namespace (default: leadpoet-prod)"
            echo "  --release-name NAME Helm release name (default: leadpoet-api)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate prerequisites
log_info "Validating prerequisites..."

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    log_error "kubectl is not installed or not in PATH"
    exit 1
fi

# Check if helm is available
if ! command -v helm &> /dev/null; then
    log_error "helm is not installed or not in PATH"
    exit 1
fi

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    log_warning "Namespace $NAMESPACE does not exist. Creating..."
    kubectl create namespace "$NAMESPACE"
    log_success "Namespace $NAMESPACE created"
fi

# Check if chart directory exists
if [[ ! -d "$CHART_PATH" ]]; then
    log_error "Chart directory $CHART_PATH does not exist"
    exit 1
fi

# Validate Helm chart
log_info "Validating Helm chart..."
if ! helm lint "$CHART_PATH"; then
    log_error "Helm chart validation failed"
    exit 1
fi

# Check for existing release
EXISTING_RELEASE=$(helm list -n "$NAMESPACE" -q | grep "^$RELEASE_NAME$" || true)

if [[ -n "$EXISTING_RELEASE" ]]; then
    log_info "Existing release found: $RELEASE_NAME"
    
    # Get current values for comparison
    CURRENT_VALUES=$(helm get values "$RELEASE_NAME" -n "$NAMESPACE" -o yaml)
    
    # Prepare upgrade command
    UPGRADE_CMD="helm upgrade $RELEASE_NAME $CHART_PATH -n $NAMESPACE"
    
    if [[ -n "$VALUES_FILE" ]]; then
        UPGRADE_CMD="$UPGRADE_CMD -f $VALUES_FILE"
    fi
    
    # Add dry-run flag for validation
    UPGRADE_CMD="$UPGRADE_CMD --dry-run"
    
    log_info "Performing dry-run upgrade..."
    if ! eval "$UPGRADE_CMD"; then
        log_error "Dry-run upgrade failed"
        exit 1
    fi
    
    # Perform actual upgrade
    log_info "Performing upgrade..."
    UPGRADE_CMD="helm upgrade $RELEASE_NAME $CHART_PATH -n $NAMESPACE"
    if [[ -n "$VALUES_FILE" ]]; then
        UPGRADE_CMD="$UPGRADE_CMD -f $VALUES_FILE"
    fi
    
    if ! eval "$UPGRADE_CMD"; then
        log_error "Upgrade failed"
        log_info "Rolling back to previous version..."
        helm rollback "$RELEASE_NAME" -n "$NAMESPACE"
        exit 1
    fi
    
    log_success "Upgrade completed successfully"
else
    log_info "No existing release found. Performing fresh installation..."
    
    # Prepare install command
    INSTALL_CMD="helm install $RELEASE_NAME $CHART_PATH -n $NAMESPACE"
    
    if [[ -n "$VALUES_FILE" ]]; then
        INSTALL_CMD="$INSTALL_CMD -f $VALUES_FILE"
    fi
    
    # Add dry-run flag for validation
    INSTALL_CMD="$INSTALL_CMD --dry-run"
    
    log_info "Performing dry-run installation..."
    if ! eval "$INSTALL_CMD"; then
        log_error "Dry-run installation failed"
        exit 1
    fi
    
    # Perform actual installation
    log_info "Performing installation..."
    INSTALL_CMD="helm install $RELEASE_NAME $CHART_PATH -n $NAMESPACE"
    if [[ -n "$VALUES_FILE" ]]; then
        INSTALL_CMD="$INSTALL_CMD -f $VALUES_FILE"
    fi
    
    if ! eval "$INSTALL_CMD"; then
        log_error "Installation failed"
        exit 1
    fi
    
    log_success "Installation completed successfully"
fi

# Wait for deployment to be ready
log_info "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/"$RELEASE_NAME" -n "$NAMESPACE"

# Check pod status
log_info "Checking pod status..."
kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=$CHART_NAME"

# Check service status
log_info "Checking service status..."
kubectl get svc -n "$NAMESPACE" -l "app.kubernetes.io/name=$CHART_NAME"

# Check HPA status
log_info "Checking HPA status..."
kubectl get hpa -n "$NAMESPACE" -l "app.kubernetes.io/name=$CHART_NAME" 2>/dev/null || log_warning "No HPA found"

# Health check
log_info "Performing health check..."

# Try to get load balancer IP first, then fallback to hostname
SERVICE_IP=$(kubectl get svc "$RELEASE_NAME" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
SERVICE_HOSTNAME=$(kubectl get svc "$RELEASE_NAME" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")

if [[ -n "$SERVICE_IP" ]]; then
    log_info "Using load balancer IP: $SERVICE_IP"
    if curl -f "http://$SERVICE_IP/healthz" &>/dev/null; then
        log_success "Health check passed"
    else
        log_warning "Health check failed (service may still be starting)"
    fi
elif [[ -n "$SERVICE_HOSTNAME" ]]; then
    log_info "Using load balancer hostname: $SERVICE_HOSTNAME"
    if curl -f "http://$SERVICE_HOSTNAME/healthz" &>/dev/null; then
        log_success "Health check passed"
    else
        log_warning "Health check failed (service may still be starting)"
    fi
else
    log_info "Load balancer IP/hostname not available yet (may be using ClusterIP or still provisioning)"
fi

log_success "Deployment completed successfully!"
log_info "Release: $RELEASE_NAME"
log_info "Namespace: $NAMESPACE"
log_info "Chart: $CHART_PATH" 