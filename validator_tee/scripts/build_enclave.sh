#!/bin/bash
#
# Build Validator Nitro Enclave Image
# ====================================
# This script builds the validator enclave Docker image and converts it to .eif format
#
# TWO-STAGE REPRODUCIBILITY:
# 1. Base image (Dockerfile.base) - built ONCE with yum install, cached
# 2. Enclave image (Dockerfile.enclave) - uses base, only COPY operations
# 3. Post-build normalization - all timestamps set to epoch 0
# 4. EIF build from normalized image → Reproducible PCR0!
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR_TEE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$VALIDATOR_TEE_DIR")"

echo "=========================================="
echo "🔨 Building Validator Nitro Enclave Image"
echo "=========================================="
echo ""
echo "Script dir: $SCRIPT_DIR"
echo "Validator TEE dir: $VALIDATOR_TEE_DIR"
echo "Repo root: $REPO_ROOT"
echo ""

# Step 0: Normalize file permissions for reproducibility
# Docker COPY includes file permissions in layer hash
# Different machines may have different umask settings (644 vs 664)
# We normalize ALL files to 644 to ensure identical layer hashes
echo "🔧 Step 0: Normalizing file permissions..."
cd "$REPO_ROOT"

# Normalize permissions on all files that will be copied into the enclave
chmod 644 validator_tee/enclave/requirements.txt 2>/dev/null || true
chmod 644 validator_tee/enclave/*.py 2>/dev/null || true
chmod 644 leadpoet_canonical/*.py 2>/dev/null || true
chmod 644 neurons/validator.py 2>/dev/null || true
chmod 644 validator_models/automated_checks.py 2>/dev/null || true

# Also normalize any other Python files that might be included
find validator_tee/enclave -type f -name "*.py" -exec chmod 644 {} \; 2>/dev/null || true
find validator_tee/enclave -type f -name "*.txt" -exec chmod 644 {} \; 2>/dev/null || true
find leadpoet_canonical -type f -name "*.py" -exec chmod 644 {} \; 2>/dev/null || true

echo "   ✓ File permissions normalized to 644"

# Step 1: Ensure base image exists (built once, cached)
echo ""
echo "📦 Step 1: Checking base image..."
if ! docker images -q validator-base:v1 | grep -q .; then
    echo "   Building base image (one-time operation)..."
    docker build \
        -f "$VALIDATOR_TEE_DIR/Dockerfile.base" \
        -t validator-base:v1 \
        "$REPO_ROOT"
    echo "   ✓ Base image built"
else
    echo "   ✓ Base image already exists"
fi

# Step 2: Build enclave Docker image
echo ""
echo "📦 Step 2: Building enclave Docker image..."
echo "   Build context: $REPO_ROOT"
echo "   Dockerfile: $VALIDATOR_TEE_DIR/Dockerfile.enclave"

# Build with --no-cache for code layers (base image is cached)
docker build --no-cache \
    -f "$VALIDATOR_TEE_DIR/Dockerfile.enclave" \
    -t validator-tee-enclave:raw \
    "$REPO_ROOT"

# Step 3: NORMALIZE the image for reproducible PCR0
echo ""
echo "🔄 Step 3: Normalizing image timestamps for reproducibility..."

python3 << 'NORMALIZE_SCRIPT'
import json
import tarfile
import hashlib
import os
import shutil
from pathlib import Path
import tempfile

def normalize_docker_image(image_name, normalized_name):
    """Normalize Docker image timestamps for reproducible PCR0."""
    work_dir = Path(tempfile.mkdtemp(prefix="pcr0_normalize_"))
    
    try:
        # Export image
        print(f"   Exporting {image_name}...")
        os.system(f"docker save {image_name} -o {work_dir}/orig.tar")
        
        # Extract
        with tarfile.open(f"{work_dir}/orig.tar", "r") as tar:
            tar.extractall(work_dir)
        
        # Read manifest
        with open(work_dir / "manifest.json") as f:
            manifest = json.load(f)
        
        layers = manifest[0]["Layers"]
        config_path = manifest[0]["Config"]
        
        print(f"   Normalizing {len(layers)} layers...")
        
        # Process each layer - normalize timestamps AND file order
        new_layers = []
        for layer_path in layers:
            full_path = work_dir / layer_path
            norm_path = str(full_path) + ".norm"
            
            # Rewrite tar with all timestamps = 0 AND sorted file order
            # Sorting ensures identical layers regardless of yum install order
            with tarfile.open(str(full_path), "r") as old_tar:
                with tarfile.open(norm_path, "w") as new_tar:
                    # Sort members alphabetically by name for deterministic order
                    members = sorted(old_tar.getmembers(), key=lambda m: m.name)
                    for member in members:
                        member.mtime = 0
                        if member.isfile():
                            content = old_tar.extractfile(member)
                            new_tar.addfile(member, content)
                        else:
                            new_tar.addfile(member)
            
            # Compute new hash
            h = hashlib.sha256()
            with open(norm_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            new_hash = h.hexdigest()
            
            new_layer_name = "blobs/sha256/" + new_hash
            new_layer_full = work_dir / new_layer_name
            new_layer_full.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(norm_path, new_layer_full)
            
            if str(full_path) != str(new_layer_full):
                try:
                    os.remove(full_path)
                except:
                    pass
            
            new_layers.append(new_layer_name)
        
        # Normalize config
        with open(work_dir / config_path) as f:
            config = json.load(f)
        
        config["created"] = "1970-01-01T00:00:00Z"
        
        new_diff_ids = []
        for layer in new_layers:
            layer_hash = layer.split("/")[-1]
            new_diff_ids.append("sha256:" + layer_hash)
        config["rootfs"]["diff_ids"] = new_diff_ids
        
        if "history" in config:
            for h in config["history"]:
                if "created" in h:
                    h["created"] = "1970-01-01T00:00:00Z"
        
        config_json = json.dumps(config, separators=(",", ":"))
        new_config_hash = hashlib.sha256(config_json.encode()).hexdigest()
        new_config_path = work_dir / "blobs" / "sha256" / new_config_hash
        new_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(new_config_path, "w") as f:
            f.write(config_json)
        
        try:
            os.remove(work_dir / config_path)
        except:
            pass
        
        manifest[0]["Layers"] = new_layers
        manifest[0]["Config"] = "blobs/sha256/" + new_config_hash
        manifest[0]["RepoTags"] = [normalized_name]
        
        with open(work_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)
        
        # Update index.json if it exists (OCI format requires this)
        index_path = work_dir / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            for m in index.get("manifests", []):
                if m.get("digest", "").startswith("sha256:"):
                    m["digest"] = "sha256:" + new_config_hash
            with open(index_path, "w") as f:
                json.dump(index, f)
            print("   Updated index.json for OCI format")
        
        # Create normalized tar
        with tarfile.open(f"{work_dir}/normalized.tar", "w") as tar:
            for item in work_dir.iterdir():
                if item.name not in ["orig.tar", "normalized.tar"]:
                    tar.add(item, arcname=item.name)
        
        # Load normalized image
        print(f"   Loading normalized image as {normalized_name}...")
        os.system(f"docker load -i {work_dir}/normalized.tar 2>/dev/null")
        os.system(f"docker tag sha256:{new_config_hash} {normalized_name} 2>/dev/null")
        
        print(f"   ✓ Image normalized successfully")
        return True
        
    except Exception as e:
        print(f"   ✗ Normalization failed: {e}")
        return False
    finally:
        shutil.rmtree(work_dir)

normalize_docker_image("validator-tee-enclave:raw", "validator-tee-enclave:latest")
NORMALIZE_SCRIPT

# Cleanup raw image
docker rmi validator-tee-enclave:raw 2>/dev/null || true

# Step 4: Build enclave image file (.eif) from NORMALIZED image
echo ""
echo "🔐 Step 4: Building enclave image file (.eif)..."

cd "$VALIDATOR_TEE_DIR"
nitro-cli build-enclave \
    --docker-uri validator-tee-enclave:latest \
    --output-file validator-enclave.eif \
    | tee enclave_build_output.txt

# Step 5: Extract measurements
echo ""
echo "📊 Step 5: Extracting enclave measurements..."
echo ""
echo "✅ Validator enclave built successfully!"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "IMPORTANT - SAVE THESE VALUES:"
echo "═══════════════════════════════════════════════════════════"
grep -E "PCR0|PCR1|PCR2" enclave_build_output.txt || echo "(PCR values not found)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Run enclave: bash scripts/start_enclave.sh"
echo "  2. Check status: nitro-cli describe-enclaves"
echo "  3. View logs: nitro-cli console --enclave-id <ID>"
echo ""
