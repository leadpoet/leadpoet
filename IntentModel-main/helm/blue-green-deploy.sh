#!/bin/bash

# Leadpoet Blue-Green Deployment Script
# Implements blue-green deployment strategy for zero-downtime deployments

set -euo pipefail

# Configuration
NAMESPACE="leadpoet-prod"
SERVICE_NAME="leadpoet-api"
BLUE_LABEL="blue"
GREEN_LABEL="green"
TRAFFIC_SPLIT_PERCENT=100
HEALTH_CHECK_ENDPOINT="/healthz"
HEALTH_CHECK_TIMEOUT=30

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
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --service-name)
            SERVICE_NAME="$2"
            shift 2
            ;;
        --traffic-split)
            TRAFFIC_SPLIT_PERCENT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --namespace NAME        Kubernetes namespace (default: leadpoet-prod)"
            echo "  --service-name NAME     Service name (default: leadpoet-api)"
            echo "  --traffic-split PERCENT Traffic split percentage (default: 100)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get current active environment
get_active_environment() {
    local service_selector=$(kubectl get svc "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.selector.environment}' 2>/dev/null || echo "")
    if [[ "$service_selector" == "$BLUE_LABEL" ]]; then
        echo "$BLUE_LABEL"
    elif [[ "$service_selector" == "$GREEN_LABEL" ]]; then
        echo "$GREEN_LABEL"
    else
        echo ""
    fi
}

# Get inactive environment
get_inactive_environment() {
    local active_env=$(get_active_environment)
    if [[ "$active_env" == "$BLUE_LABEL" ]]; then
        echo "$GREEN_LABEL"
    elif [[ "$active_env" == "$GREEN_LABEL" ]]; then
        echo "$BLUE_LABEL"
    else
        echo "$BLUE_LABEL"  # Default to blue if no active environment
    fi
}

# Deploy to environment
deploy_to_environment() {
    local environment="$1"
    local chart_path="$2"
    local values_file="$3"
    
    log_info "Deploying to $environment environment..."
    
    local release_name="${SERVICE_NAME}-${environment}"
    local install_cmd="helm upgrade --install $release_name $chart_path -n $NAMESPACE"
    
    if [[ -n "$values_file" ]]; then
        install_cmd="$install_cmd -f $values_file"
    fi
    
    # Add environment-specific values
    install_cmd="$install_cmd --set deployment.labels.environment=$environment"
    
    # Perform deployment
    if ! eval "$install_cmd"; then
        log_error "Deployment to $environment failed"
        return 1
    fi
    
    # Wait for deployment to be ready
    log_info "Waiting for $environment deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/"$release_name" -n "$NAMESPACE"
    
    log_success "Deployment to $environment completed"
    return 0
}

# Health check for environment
health_check_environment() {
    local environment="$1"
    local release_name="${SERVICE_NAME}-${environment}"
    
    log_info "Performing health check for $environment environment..."
    
    # Get service port
    local service_port=$(kubectl get svc "$release_name" -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "")
    if [[ -z "$service_port" ]]; then
        log_error "Could not get service port for $release_name"
        return 1
    fi
    
    # Port forward to service for health check
    local port_forward_pid=""
    kubectl port-forward svc/"$release_name" "$service_port:$service_port" -n "$NAMESPACE" >/dev/null 2>&1 &
    port_forward_pid=$!
    
    # Wait for port forward to be ready
    sleep 5
    
    # Perform health check
    local health_check_result=0
    if curl -f "http://localhost:$service_port$HEALTH_CHECK_ENDPOINT" --max-time "$HEALTH_CHECK_TIMEOUT" >/dev/null 2>&1; then
        log_success "Health check passed for $environment environment"
        health_check_result=0
    else
        log_error "Health check failed for $environment environment"
        health_check_result=1
    fi
    
    # Kill port forward
    kill "$port_forward_pid" 2>/dev/null || true
    
    return "$health_check_result"
}

# Switch traffic to environment
switch_traffic() {
    local target_environment="$1"
    local traffic_split_percent="${2:-100}"
    
    log_info "Switching traffic to $target_environment environment (${traffic_split_percent}%)..."
    
    if [[ "$traffic_split_percent" -eq 100 ]]; then
        # Full traffic switch - update service selector
        kubectl patch svc "$SERVICE_NAME" -n "$NAMESPACE" -p "{\"spec\":{\"selector\":{\"environment\":\"$target_environment\"}}}"
        log_success "Full traffic switch to $target_environment environment"
    else
        # Partial traffic split - implement using Istio VirtualService or similar
        log_warning "Partial traffic splitting (${traffic_split_percent}%) requires Istio or similar service mesh"
        log_info "For now, performing full traffic switch. Consider implementing Istio VirtualService for partial splits."
        
        # Fallback to full switch for now
        kubectl patch svc "$SERVICE_NAME" -n "$NAMESPACE" -p "{\"spec\":{\"selector\":{\"environment\":\"$target_environment\"}}}"
        log_success "Traffic switched to $target_environment environment (fallback to full switch)"
    fi
}

# Clean up old environment
cleanup_environment() {
    local environment="$1"
    local release_name="${SERVICE_NAME}-${environment}"
    
    log_info "Cleaning up $environment environment..."
    
    # Delete the old deployment
    if helm list -n "$NAMESPACE" | grep -q "$release_name"; then
        helm uninstall "$release_name" -n "$NAMESPACE"
        log_success "Cleaned up $environment environment"
    else
        log_warning "No release found for $environment environment"
    fi
}

# Main deployment logic
main() {
    log_info "Starting blue-green deployment..."
    log_info "Traffic split percentage: ${TRAFFIC_SPLIT_PERCENT}%"
    
    # Validate traffic split percentage
    if [[ "$TRAFFIC_SPLIT_PERCENT" -lt 0 || "$TRAFFIC_SPLIT_PERCENT" -gt 100 ]]; then
        log_error "Traffic split percentage must be between 0 and 100"
        exit 1
    fi
    
    # Check prerequisites
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed or not in PATH"
        exit 1
    fi
    
    # Get current active environment
    local active_env=$(get_active_environment)
    local target_env=$(get_inactive_environment)
    
    log_info "Current active environment: ${active_env:-none}"
    log_info "Target environment: $target_env"
    
    # Deploy to target environment
    if ! deploy_to_environment "$target_env" "./helm/leadpoet-api" ""; then
        log_error "Deployment to $target_env failed"
        exit 1
    fi
    
    # Health check target environment
    if ! health_check_environment "$target_env"; then
        log_error "Health check failed for $target_env environment"
        log_info "Rolling back deployment..."
        helm uninstall "${SERVICE_NAME}-${target_env}" -n "$NAMESPACE"
        exit 1
    fi
    
    # Switch traffic to target environment
    switch_traffic "$target_env" "$TRAFFIC_SPLIT_PERCENT"
    
    # Verify traffic switch
    log_info "Verifying traffic switch..."
    sleep 10
    
    if ! health_check_environment "$target_env"; then
        log_error "Health check failed after traffic switch"
        log_info "Rolling back to previous environment..."
        if [[ -n "$active_env" ]]; then
            switch_traffic "$active_env" "100"
        fi
        exit 1
    fi
    
    # Clean up old environment if it exists and we did a full switch
    if [[ -n "$active_env" && "$TRAFFIC_SPLIT_PERCENT" -eq 100 ]]; then
        cleanup_environment "$active_env"
    elif [[ -n "$active_env" && "$TRAFFIC_SPLIT_PERCENT" -lt 100 ]]; then
        log_info "Keeping $active_env environment for partial traffic split"
    fi
    
    log_success "Blue-green deployment completed successfully!"
    log_info "Active environment: $target_env"
    log_info "Traffic split: ${TRAFFIC_SPLIT_PERCENT}% to $target_env"
}

# Run main function
main "$@" 