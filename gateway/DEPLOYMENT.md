# Gateway Deployment Guide

**⚠️ FOR SUBNET OWNERS ONLY - Miners/validators do NOT need this**

This guide is for deploying the Leadpoet gateway service. Miners and validators only need the verification scripts in `scripts/`.

---

## Prerequisites

- Docker & Docker Compose installed
- AWS EC2 instance with Nitro Enclaves enabled
- Arweave wallet with sufficient balance (~3 AR)
- Supabase project with proper schema

---

## Setup

### 1. Clone Repository (Private Server Only)

```bash
git clone https://github.com/leadpoet/Leadpoet.git
cd Leadpoet
```

### 2. Create Environment File

Create `.env` in the project root:

```bash
# MinIO Storage Credentials (choose strong passwords!)
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=<your-secure-password-32-chars-minimum>

# Gateway Public IP (for MinIO presigned URLs)
GATEWAY_PUBLIC_IP=<your-ec2-public-ip>

# Supabase Configuration
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<your-service-role-key>

# Build Information
BUILD_ID=production
GITHUB_SHA=$(git rev-parse HEAD)
```

**Security Notes:**
- ✅ Use strong, unique passwords (32+ characters)
- ✅ Never commit `.env` to git (already in `.gitignore`)
- ✅ Restrict file permissions: `chmod 600 .env`

### 3. Copy Docker Compose Templates

```bash
# Copy templates to actual files
cp docker-compose.gateway.yml.example docker-compose.gateway.yml
cp docker-compose.minio.yml.example docker-compose.minio.yml

# Note: These files are gitignored for security
```

### 4. Setup Arweave Keyfile

```bash
# Create secrets directory
mkdir -p gateway/secrets/

# Copy your Arweave keyfile
cp /path/to/your/arweave_keyfile.json gateway/secrets/

# Secure permissions
chmod 600 gateway/secrets/arweave_keyfile.json
```

### 5. Deploy Gateway

```bash
# Build and start services
docker compose -f docker-compose.gateway.yml up -d --build

# View logs
docker compose -f docker-compose.gateway.yml logs -f gateway

# Check health
curl http://localhost:8000/health
```

### 6. Deploy TEE Enclave (On EC2)

See `gateway/tee/SECURITY.md` for TEE enclave deployment instructions.

```bash
# On EC2 instance
cd gateway/tee
./build_enclave.sh
./start_enclave.sh

# Verify enclave is running
nitro-cli describe-enclaves
```

---

## Verification

### Test Gateway Endpoints

```bash
# Health check
curl http://<your-ip>:8000/health

# Attestation (TEE proof)
curl http://<your-ip>:8000/attest

# Manifest (epoch assignments)
curl http://<your-ip>:8000/manifest
```

### Verify Attestation

```bash
# From any machine (miners/validators can do this)
python scripts/verify_attestation.py http://<your-ip>:8000
```

---

## Monitoring

### View Logs

```bash
# Gateway logs
docker compose logs -f gateway

# MinIO logs
docker compose logs -f minio

# TEE enclave logs
nitro-cli console --enclave-id $(nitro-cli describe-enclaves | jq -r '.[0].EnclaveID')
```

### Check Arweave Balance

```bash
# Install arweave CLI
npm install -g arweave

# Check balance
arweave balance gateway/secrets/arweave_keyfile.json
```

**Alert**: Set up monitoring for low balance (< 1 AR)

---

## Security Checklist

- [ ] Strong MinIO credentials (32+ chars)
- [ ] `.env` file secured (`chmod 600`)
- [ ] Arweave keyfile secured (`chmod 600`)
- [ ] Supabase service key not exposed publicly
- [ ] Docker compose files not committed to git
- [ ] TEE enclave attestation verified
- [ ] PCR0 published for miners/validators to verify
- [ ] Firewall configured (only ports 8000, 9000, 9001)
- [ ] SSL/TLS enabled (use nginx reverse proxy)

---

## Updating Gateway

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker compose -f docker-compose.gateway.yml down
docker compose -f docker-compose.gateway.yml up -d --build

# Verify new attestation
curl http://localhost:8000/attest

# Publish new PCR0 for miners/validators
```

---

## Troubleshooting

### MinIO Connection Failed

```bash
# Check MinIO is running
docker compose ps

# Check MinIO logs
docker compose logs minio

# Test MinIO access
curl http://localhost:9000/minio/health/live
```

### Arweave Upload Failures

```bash
# Check balance
arweave balance gateway/secrets/arweave_keyfile.json

# Check keyfile permissions
ls -la gateway/secrets/arweave_keyfile.json

# View gateway logs for Arweave errors
docker compose logs gateway | grep -i arweave
```

### TEE Enclave Not Starting

```bash
# Check enclave status
nitro-cli describe-enclaves

# View enclave logs
nitro-cli console --enclave-id <id>

# Rebuild enclave
cd gateway/tee
./stop_enclave.sh
./build_enclave.sh --no-cache
./start_enclave.sh
```

---

## Important Notes

### For Miners/Validators

**You do NOT need any of this deployment information!**

Miners and validators only need:
- ✅ `scripts/verify_attestation.py` - Verify gateway is running canonical code
- ✅ `scripts/verify_code_hash.py` - Verify code matches GitHub
- ✅ `scripts/verify_merkle_inclusion.py` - Verify event inclusion
- ✅ `scripts/VERIFICATION_GUIDE.md` - Step-by-step verification instructions

The gateway is trustless - you can verify it's running the correct code without needing deployment access.

---

## Support

For gateway operator support:
- Email: hello@leadpoet.com
- Discord: @leadpoet (Bittensor Discord)

