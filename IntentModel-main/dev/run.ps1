#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Leadpoet Development Environment Setup and Run Script
    
.DESCRIPTION
    PowerShell 7 script that sets up the Leadpoet development environment
    and runs the application. Mirrors the functionality of run.sh for
    cross-platform compatibility.
    
.PARAMETER Environment
    Target environment (local, staging, prod). Default: local
    
.PARAMETER SkipSetup
    Skip environment setup and run directly
    
.EXAMPLE
    .\run.ps1
    .\run.ps1 -Environment staging
    .\run.ps1 -SkipSetup
#>

param(
    [string]$Environment = "local",
    [switch]$SkipSetup
)

# Script configuration
$ErrorActionPreference = "Stop"
$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $PSScriptRoot

# Colors for output
$Colors = @{
    Info = "Cyan"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Colors[$Color]
}

function Test-Prerequisites {
    Write-ColorOutput "Checking prerequisites..." "Info"
    
    $prerequisites = @{
        "Python" = "python --version"
        "pip" = "pip --version"
        "Docker" = "docker --version"
        "kubectl" = "kubectl version --client"
        "AWS CLI" = "aws --version"
    }
    
    $missing = @()
    
    foreach ($tool in $prerequisites.GetEnumerator()) {
        try {
            $null = Invoke-Expression $tool.Value 2>$null
            Write-ColorOutput "✓ $($tool.Key) found" "Success"
        }
        catch {
            Write-ColorOutput "✗ $($tool.Key) not found" "Warning"
            $missing += $tool.Key
        }
    }
    
    if ($missing.Count -gt 0) {
        Write-ColorOutput "Missing prerequisites: $($missing -join ', ')" "Error"
        Write-ColorOutput "Please install missing tools and try again." "Error"
        exit 1
    }
}

function Set-EnvironmentVariables {
    Write-ColorOutput "Setting up environment variables..." "Info"
    
    # Load .env file if it exists
    $envFile = Join-Path $ProjectRoot ".env"
    if (Test-Path $envFile) {
        Get-Content $envFile | ForEach-Object {
            if ($_ -match '^([^#][^=]+)=(.*)$') {
                $name = $matches[1].Trim()
                $value = $matches[2].Trim()
                [Environment]::SetEnvironmentVariable($name, $value, "Process")
                Write-ColorOutput "Set $name" "Success"
            }
        }
    }
    else {
        Write-ColorOutput "No .env file found, using defaults" "Warning"
    }
    
    # Set default values if not present
    $defaults = @{
        "APP_ENV" = $Environment
        "LOG_LEVEL" = "INFO"
        "DB_HOST" = "localhost"
        "DB_PORT" = "5432"
        "REDIS_HOST" = "localhost"
        "REDIS_PORT" = "6379"
    }
    
    foreach ($default in $defaults.GetEnumerator()) {
        if (-not [Environment]::GetEnvironmentVariable($default.Key)) {
            [Environment]::SetEnvironmentVariable($default.Key, $default.Value, "Process")
        }
    }
}

function Install-Dependencies {
    Write-ColorOutput "Installing Python dependencies..." "Info"
    
    $requirementsFile = Join-Path $ProjectRoot "requirements.txt"
    if (Test-Path $requirementsFile) {
        try {
            pip install -r $requirementsFile
            Write-ColorOutput "✓ Dependencies installed" "Success"
        }
        catch {
            Write-ColorOutput "✗ Failed to install dependencies" "Error"
            exit 1
        }
    }
    else {
        Write-ColorOutput "No requirements.txt found" "Warning"
    }
}

function Start-Database {
    Write-ColorOutput "Starting database services..." "Info"
    
    # Check if Docker is running
    try {
        $null = docker info 2>$null
    }
    catch {
        Write-ColorOutput "Docker is not running. Please start Docker Desktop." "Error"
        exit 1
    }
    
    # Start PostgreSQL with TimescaleDB
    # Get database password from environment or use default
    $dbPassword = if ($env:DB_PASSWORD) { $env:DB_PASSWORD } else { "leadpoet123" }

    $postgresContainer = "leadpoet-postgres"
    if (-not (docker ps -q -f name=$postgresContainer)) {
        Write-ColorOutput "Starting PostgreSQL with TimescaleDB..." "Info"
        docker run -d `
            --name $postgresContainer `
            -e POSTGRES_DB=leadpoet `
            -e POSTGRES_USER=leadpoet `
            -e POSTGRES_PASSWORD=$dbPassword `
            -p 5432:5432 `
            timescale/timescaledb:latest-pg16
    }    }
    else {
        Write-ColorOutput "✓ PostgreSQL already running" "Success"
    }
    
    # Start Redis
    $redisContainer = "leadpoet-redis"
    if (-not (docker ps -q -f name=$redisContainer)) {
        Write-ColorOutput "Starting Redis..." "Info"
        docker run -d `
            --name $redisContainer `
            -p 6379:6379 `
            redis:7-alpine
    }
    else {
        Write-ColorOutput "✓ Redis already running" "Success"
    }
    
    # Wait for services to be ready
    Write-ColorOutput "Waiting for services to be ready..." "Info"
    Start-Sleep -Seconds 10
}

function Run-Migrations {
    Write-ColorOutput "Running database migrations..." "Info"
    
    try {
        # Check if alembic is available
        $alembicPath = Join-Path $ProjectRoot "alembic.ini"
        if (Test-Path $alembicPath) {
            python -m alembic upgrade head
            Write-ColorOutput "✓ Migrations completed" "Success"
        }
        else {
            Write-ColorOutput "No alembic.ini found, skipping migrations" "Warning"
        }
    }
    catch {
        Write-ColorOutput "✗ Migration failed: $($_.Exception.Message)" "Error"
        exit 1
    }
}

function Start-Application {
    Write-ColorOutput "Starting Leadpoet application..." "Info"
    
    $appPath = Join-Path $ProjectRoot "app"
    if (Test-Path $appPath) {
        try {
            # Change to app directory
            Push-Location $appPath
            
            # Start FastAPI server
            Write-ColorOutput "Starting FastAPI server on http://localhost:8000" "Info"
            uvicorn main:app --reload --host 0.0.0.0 --port 8000
        }
        catch {
            Write-ColorOutput "✗ Failed to start application: $($_.Exception.Message)" "Error"
            exit 1
        }
        finally {
            Pop-Location
        }
    }
    else {
        Write-ColorOutput "Application directory not found. Creating stub..." "Warning"
        New-Item -ItemType Directory -Path $appPath -Force | Out-Null
        
        # Create basic FastAPI app
        $mainPy = @"
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Leadpoet API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Leadpoet Intent Model v1.1"}

@app.get("/healthz")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"@
        
        Set-Content -Path (Join-Path $appPath "main.py") -Value $mainPy
        
        Write-ColorOutput "Created stub application. Run again to start server." "Info"
    }
}

function Show-Status {
    Write-ColorOutput "`n=== Leadpoet Development Environment Status ===" "Info"
    
    # Check Docker containers
    $containers = @("leadpoet-postgres", "leadpoet-redis")
    foreach ($container in $containers) {
        if (docker ps -q -f name=$container) {
            Write-ColorOutput "✓ $container running" "Success"
        }
        else {
            Write-ColorOutput "✗ $container not running" "Error"
        }
    }
    
    # Check application
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/healthz" -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-ColorOutput "✓ Application running on http://localhost:8000" "Success"
        }
    }
    catch {
        Write-ColorOutput "✗ Application not responding" "Warning"
    }
    
    Write-ColorOutput "`nAccess points:" "Info"
    Write-ColorOutput "- API: http://localhost:8000" "Info"
    Write-ColorOutput "- Docs: http://localhost:8000/docs" "Info"
    Write-ColorOutput "- Database: localhost:5432" "Info"
    Write-ColorOutput "- Redis: localhost:6379" "Info"
}

# Main execution
try {
    Write-ColorOutput "🚀 Leadpoet Development Environment Setup" "Info"
    Write-ColorOutput "Environment: $Environment" "Info"
    
    if (-not $SkipSetup) {
        Test-Prerequisites
        Set-EnvironmentVariables
        Install-Dependencies
        Start-Database
        Run-Migrations
    }
    
    Start-Application
}
catch {
    Write-ColorOutput "Fatal error: $($_.Exception.Message)" "Error"
    exit 1
}
finally {
    Show-Status
} 