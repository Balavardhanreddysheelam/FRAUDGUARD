#!/bin/bash
# FraudGuard Deployment Script
# This script helps deploy FraudGuard to production

set -e

echo "🚀 FraudGuard Deployment Script"
echo "================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Check if .env exists
check_env_file() {
    if [ ! -f "backend/.env" ]; then
        print_warning ".env file not found in backend/"
        print_info "Creating .env from env.example..."
        if [ -f "backend/env.example" ]; then
            cp backend/env.example backend/.env
            print_success ".env file created"
            print_warning "Please update backend/.env with your configuration before continuing"
            return 1
        else
            print_error "env.example not found!"
            return 1
        fi
    else
        print_success ".env file exists"
        return 0
    fi
}

# Check if model exists
check_model() {
    if [ ! -d "inference/model/fraudguard-8b-merged" ]; then
        print_warning "Model directory not found at inference/model/fraudguard-8b-merged"
        print_info "Please ensure the model is in the correct location"
        return 1
    else
        print_success "Model directory found"
        return 0
    fi
}

# Generate API key
generate_api_key() {
    if command -v openssl &> /dev/null; then
        API_KEY=$(openssl rand -hex 32)
        print_success "Generated API key: $API_KEY"
        print_info "Add this to your .env file: API_KEY=$API_KEY"
        return 0
    else
        print_warning "openssl not found. Please generate API key manually: openssl rand -hex 32"
        return 1
    fi
}

# Start services with docker-compose
start_services() {
    print_info "Starting services with docker-compose..."
    if command -v docker-compose &> /dev/null || command -v docker &> /dev/null; then
        if command -v docker-compose &> /dev/null; then
            docker-compose up -d
        else
            docker compose up -d
        fi
        print_success "Services started"
        print_info "Waiting for services to be healthy..."
        sleep 10
        return 0
    else
        print_error "Docker not found. Please install Docker first."
        return 1
    fi
}

# Check service health
check_health() {
    print_info "Checking backend health..."
    sleep 5
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "Backend is healthy!"
        curl http://localhost:8000/health | python3 -m json.tool 2>/dev/null || curl http://localhost:8000/health
        return 0
    else
        print_warning "Backend health check failed. Services may still be starting..."
        print_info "Check logs with: docker-compose logs -f backend"
        return 1
    fi
}

# Main deployment flow
main() {
    echo ""
    print_info "Starting deployment checks..."
    echo ""
    
    # Check prerequisites
    ENV_OK=false
    MODEL_OK=false
    
    if check_env_file; then
        ENV_OK=true
    fi
    
    if check_model; then
        MODEL_OK=true
    fi
    
    echo ""
    print_info "Deployment Options:"
    echo "1. Local development (docker-compose)"
    echo "2. Generate API key"
    echo "3. Check service health"
    echo "4. View logs"
    echo "5. Exit"
    echo ""
    read -p "Select option (1-5): " option
    
    case $option in
        1)
            if [ "$ENV_OK" = true ] && [ "$MODEL_OK" = true ]; then
                start_services
                check_health
            else
                print_error "Prerequisites not met. Please fix issues above."
            fi
            ;;
        2)
            generate_api_key
            ;;
        3)
            check_health
            ;;
        4)
            print_info "Showing logs (Ctrl+C to exit)..."
            if command -v docker-compose &> /dev/null; then
                docker-compose logs -f
            else
                docker compose logs -f
            fi
            ;;
        5)
            print_info "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid option"
            ;;
    esac
}

# Run main function
main



