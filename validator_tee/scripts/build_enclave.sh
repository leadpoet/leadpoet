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
BASE_IMAGE_STAMP_FILE="$REPO_ROOT/.validator-base.dockerfile.sha256"

echo "=========================================="
echo "🔨 Building Validator Nitro Enclave Image"
echo "=========================================="
echo ""
echo "Script dir: $SCRIPT_DIR"
echo "Validator TEE dir: $VALIDATOR_TEE_DIR"
echo "Repo root: $REPO_ROOT"
echo ""

cd "$REPO_ROOT"

PCR0_COPY_PATHS=(
    "validator_tee/enclave"
    "leadpoet_canonical"
    "leadpoet_verifier"
    "research_lab"
    "gateway/research_lab"
    "gateway/qualification/utils"
    "qualification/scoring"
    "gateway/__init__.py"
    "gateway/qualification/__init__.py"
    "gateway/qualification/models.py"
    "gateway/qualification/config.py"
    "gateway/db/__init__.py"
    "gateway/db/client.py"
    "gateway/tasks/__init__.py"
    "gateway/tasks/icp_generator.py"
    "qualification/__init__.py"
    "scripts/run_research_lab_hosted_worker.py"
    "scripts/run_research_lab_hosted_worker_fleet.py"
    "scripts/run_research_lab_scoring_worker.py"
    "scripts/run_research_lab_scoring_worker_fleet.py"
    "neurons/validator.py"
    "validator_models/automated_checks.py"
)

# Step 0a: Clean PCR0 build context for reproducibility.
#
# The validator intentionally builds from this local checkout, but the Docker
# context must still equal the checked-out commit. Untracked local leftovers
# inside copied paths (.bak files, __pycache__, pyc, temp files, etc.) would be
# copied into the enclave and produce a PCR0 that the gateway's clean GitHub
# rebuild can never match.
echo "🧹 Step 0a: Cleaning PCR0 Docker context..."
git clean -ffdx -- "${PCR0_COPY_PATHS[@]}" >/dev/null 2>&1 || true
for path in "${PCR0_COPY_PATHS[@]}"; do
    if [ -d "$path" ]; then
        find "$path" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find "$path" -type f -name "*.pyc" -delete 2>/dev/null || true
        find "$path" -type f -name "*.pyo" -delete 2>/dev/null || true
    fi
done
echo "   ✓ PCR0 Docker context cleaned"

# Step 0b: Normalize file and directory permissions for reproducibility.
#
# Docker COPY preserves mode bits. A local checkout with 664 files and a sparse
# GitHub checkout with 644 files have identical content but different PCR0.
echo "🔧 Step 0b: Normalizing PCR0 Docker context permissions..."
for path in "${PCR0_COPY_PATHS[@]}"; do
    if [ -d "$path" ]; then
        find "$path" -type d -exec chmod 755 {} + 2>/dev/null || true
        find "$path" -type f -exec chmod 644 {} + 2>/dev/null || true
    elif [ -f "$path" ]; then
        chmod 644 "$path" 2>/dev/null || true
    fi
done

echo "   ✓ PCR0 Docker context permissions normalized"

echo "🔎 Step 0c: PCR0 Docker context manifest hash..."
python3 - "${PCR0_COPY_PATHS[@]}" <<'PY'
import hashlib
import os
import sys

copied_files = []
for rel_path in sys.argv[1:]:
    if os.path.isdir(rel_path):
        for root, _dirs, files in os.walk(rel_path):
            for name in files:
                copied_files.append(os.path.join(root, name))
    elif os.path.isfile(rel_path):
        copied_files.append(rel_path)

manifest_hash = hashlib.sha256()
for rel_path in sorted(set(copied_files)):
    st = os.stat(rel_path)
    manifest_hash.update(f"{st.st_mode & 0o777} {st.st_size} {rel_path}\n".encode())
    with open(rel_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            manifest_hash.update(chunk)
    manifest_hash.update(b"\n")

print(f"   manifest_sha256={manifest_hash.hexdigest()}")
PY

# Step 1: Ensure base image exists (built once, cached)
echo ""
echo "📦 Step 1: Checking base image..."
DOCKERFILE_BASE_HASH="$(python3 - "$VALIDATOR_TEE_DIR/Dockerfile.base" <<'PY'
import hashlib
import sys

with open(sys.argv[1], "rb") as f:
    print(hashlib.sha256(f.read()).hexdigest()[:16])
PY
)"
EXISTING_BASE_STAMP=""
if [ -f "$BASE_IMAGE_STAMP_FILE" ]; then
    EXISTING_BASE_STAMP="$(tr -d '[:space:]' < "$BASE_IMAGE_STAMP_FILE")"
fi
BASE_IMAGE_EXISTS=false
if docker images -q validator-base:v1 | grep -q .; then
    BASE_IMAGE_EXISTS=true
fi

if [ "$BASE_IMAGE_EXISTS" != "true" ]; then
    echo "   Building base image (missing; hash: $DOCKERFILE_BASE_HASH)..."
    docker build --no-cache \
        -f "$VALIDATOR_TEE_DIR/Dockerfile.base" \
        -t validator-base:v1 \
        "$REPO_ROOT"
    printf "%s\n" "$DOCKERFILE_BASE_HASH" > "$BASE_IMAGE_STAMP_FILE"
    echo "   ✓ Base image built"
elif [ "$EXISTING_BASE_STAMP" != "$DOCKERFILE_BASE_HASH" ]; then
    echo "   Base image stamp stale or absent (existing: ${EXISTING_BASE_STAMP:-none}, expected: $DOCKERFILE_BASE_HASH)"
    echo "   Rebuilding base image so validator PCR0 matches gateway GitHub rebuilds..."
    docker rmi -f validator-base:v1 >/dev/null 2>&1 || true
    docker build --no-cache \
        -f "$VALIDATOR_TEE_DIR/Dockerfile.base" \
        -t validator-base:v1 \
        "$REPO_ROOT"
    printf "%s\n" "$DOCKERFILE_BASE_HASH" > "$BASE_IMAGE_STAMP_FILE"
    echo "   ✓ Base image rebuilt"
else
    echo "   ✓ Base image already exists and matches Dockerfile.base hash: $DOCKERFILE_BASE_HASH"
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
        normalized_layer_paths = {}
        for layer_path in layers:
            if layer_path in normalized_layer_paths:
                new_layers.append(normalized_layer_paths[layer_path])
                continue

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
            
            normalized_layer_paths[layer_path] = new_layer_name
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
        
        # docker save may emit OCI archive metadata on newer Docker versions.
        # The normalized archive is intentionally written as a Docker archive
        # using manifest.json only; stale OCI index entries would point at the
        # pre-normalized manifest/config and make docker load reject the tar.
        for metadata_name in ("index.json", "oci-layout"):
            metadata_path = work_dir / metadata_name
            if metadata_path.exists():
                metadata_path.unlink()
        
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

if not normalize_docker_image("validator-tee-enclave:raw", "validator-tee-enclave:latest"):
    raise SystemExit(1)
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
