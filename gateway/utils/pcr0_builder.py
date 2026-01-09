"""
Dynamic PCR0 Builder for Trustless Verification

This module runs a background task that:
1. Fetches the latest commits from GitHub
2. Checks if monitored files changed
3. Builds the validator enclave and extracts PCR0
4. Caches the results for verification

TRUSTLESSNESS:
- Gateway computes PCR0 itself (no human input)
- Subnet owner CANNOT inject fake PCR0 values
- Only code actually in GitHub can produce valid PCR0

REPRODUCIBILITY (Option 1 - Pinned Dockerfile):
- Dockerfile.enclave uses pinned base image SHA256
- requirements.txt uses pinned pip package versions
- Same code = ALWAYS same PCR0 (regardless of when built)
- No TTL needed - builds are deterministic

MONITORED FILES (changes trigger rebuild):
- validator_tee/Dockerfile.enclave
- validator_tee/enclave/*
- leadpoet_canonical/*
- neurons/validator.py
- validator_models/automated_checks.py
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# GitHub repo URL (public repo - no auth needed)
GITHUB_REPO_URL = os.environ.get(
    "GITHUB_REPO_URL",
    "https://github.com/leadpoet/leadpoet.git"
)

# Branch to track
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")

# How often to check for updates (seconds)
PCR0_CHECK_INTERVAL = int(os.environ.get("PCR0_CHECK_INTERVAL", "480"))  # 8 minutes

# How many CODE VERSIONS to cache PCR0 for
# Allows validators on older code to still be accepted
PCR0_CACHE_SIZE = int(os.environ.get("PCR0_CACHE_SIZE", "3"))

# NOTE: No TTL needed with pinned Dockerfile!
# With pinned base image + pinned pip versions, builds are DETERMINISTIC.
# Same code = same PCR0, regardless of when it's built.
# Security patches require manually updating the Dockerfile pins.

# Files that affect PCR0 (if any of these change, rebuild)
MONITORED_FILES: Set[str] = {
    ".dockerignore",  # Affects Docker build context
    "validator_tee/Dockerfile.base",  # Base image definition
    "validator_tee/Dockerfile.enclave",
    "validator_tee/enclave/requirements.txt",
    "validator_tee/enclave/__init__.py",
    "validator_tee/enclave/nsm_lib.py",
    "validator_tee/enclave/tee_service.py",
    "neurons/validator.py",
    "validator_models/automated_checks.py",
}

# Directories where any file change triggers rebuild
MONITORED_DIRS: Set[str] = {
    "leadpoet_canonical/",
}

# Working directory for builds
BUILD_DIR = os.environ.get("PCR0_BUILD_DIR", "/tmp/pcr0_builder")


# =============================================================================
# Cache
# =============================================================================

# Cache structure (keyed by CONTENT HASH, not commit hash):
# {content_hash: {"pcr0": "...", "content_hash": "...", "commit_hash": "...", "built_at": timestamp}}
# This means: same code content = same cache key, regardless of commits
_pcr0_cache: Dict[str, Dict] = {}
_cache_lock = asyncio.Lock()

# Is a build currently running?
_build_in_progress = False


def get_cached_pcr0_values() -> List[str]:
    """Get all cached PCR0 values (for verification)."""
    return [entry["pcr0"] for entry in _pcr0_cache.values()]


def is_pcr0_valid(pcr0: str) -> bool:
    """Check if a PCR0 value is in our computed cache."""
    return pcr0 in get_cached_pcr0_values()


def get_cache_status() -> Dict:
    """Get current cache status for debugging."""
    return {
        "cached_content_hashes": list(_pcr0_cache.keys()),
        "cached_pcr0s": get_cached_pcr0_values(),
        "cache_entries": [
            {
                "content_hash": k,
                "commit_hash": v.get("commit_hash", "?")[:8],
                "pcr0": v["pcr0"][:32] + "...",
                "built_at": v.get("built_at"),
            }
            for k, v in _pcr0_cache.items()
        ],
        "build_in_progress": _build_in_progress,
        "cache_size": len(_pcr0_cache),
    }


# =============================================================================
# Git Operations
# =============================================================================

async def get_latest_commits(repo_dir: str, count: int = 3) -> List[Dict]:
    """Get the latest N commits from the repo."""
    proc = await asyncio.create_subprocess_exec(
        "git", "log", f"-{count}", "--format=%H|%s|%ai",
        cwd=repo_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    
    if proc.returncode != 0:
        logger.error(f"[PCR0] git log failed: {stderr.decode()}")
        return []
    
    commits = []
    for line in stdout.decode().strip().split("\n"):
        if "|" in line:
            parts = line.split("|", 2)
            commits.append({
                "hash": parts[0],
                "message": parts[1] if len(parts) > 1 else "",
                "date": parts[2] if len(parts) > 2 else "",
            })
    
    return commits


async def clone_or_update_repo(repo_dir: str) -> bool:
    """
    Clone or update the repo using sparse checkout.
    
    OPTIMIZATION: Only fetches the files needed for PCR0 verification:
    - validator_tee/ (Dockerfile and enclave code)
    - leadpoet_canonical/ (canonical modules)
    - neurons/validator.py
    - validator_models/automated_checks.py
    
    This reduces clone size from ~50MB to ~5MB and time from ~10s to ~2s.
    """
    # Environment to prevent git from prompting for credentials
    git_env = os.environ.copy()
    git_env["GIT_TERMINAL_PROMPT"] = "0"  # Don't prompt for credentials
    
    # Sparse checkout paths - only what's needed for PCR0
    # NOTE: Leading slashes are required in --no-cone mode for single files
    sparse_paths = [
        "/.dockerignore",  # Critical for reproducible builds
        "/validator_tee/",  # Includes both Dockerfile.base and Dockerfile.enclave
        "/leadpoet_canonical/",
        "/neurons/validator.py",
        "/validator_models/automated_checks.py",
    ]
    
    if os.path.exists(os.path.join(repo_dir, ".git")):
        # Update existing repo - just fetch and reset
        proc = await asyncio.create_subprocess_exec(
            "git", "fetch", "--depth", "1", "origin", GITHUB_BRANCH,
            cwd=repo_dir,
            env=git_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            logger.warning(f"[PCR0] git fetch failed: {stderr.decode()}, will retry with fresh clone")
            # Remove and re-clone
            shutil.rmtree(repo_dir)
            return await clone_or_update_repo(repo_dir)
        
        proc = await asyncio.create_subprocess_exec(
            "git", "reset", "--hard", f"origin/{GITHUB_BRANCH}",
            cwd=repo_dir,
            env=git_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            logger.error(f"[PCR0] git reset failed: {stderr.decode()}")
            return False
            
        logger.info("[PCR0] Repo updated via fetch")
    else:
        # Fresh clone with sparse checkout (minimal download)
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        os.makedirs(repo_dir, exist_ok=True)
        
        logger.info(f"[PCR0] Sparse cloning {GITHUB_REPO_URL}...")
        
        # Step 1: Clone with sparse checkout enabled (downloads only .git metadata)
        proc = await asyncio.create_subprocess_exec(
            "git", "clone",
            "--depth", "1",           # Only latest commit
            "--filter=blob:none",     # Don't download any files yet
            "--sparse",               # Enable sparse checkout
            "-b", GITHUB_BRANCH,
            GITHUB_REPO_URL, repo_dir,
            env=git_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            logger.error(f"[PCR0] git clone failed: {stderr.decode()}")
            return False
        
        # Step 2: Configure sparse checkout to only get PCR0-relevant files
        # NOTE: --no-cone is required to allow individual file paths (not just directories)
        proc = await asyncio.create_subprocess_exec(
            "git", "sparse-checkout", "set", "--no-cone", *sparse_paths,
            cwd=repo_dir,
            env=git_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            logger.error(f"[PCR0] git sparse-checkout failed: {stderr.decode()}")
            return False
        
        logger.info(f"[PCR0] Sparse clone successful (only PCR0 files: {len(sparse_paths)} paths)")
    
    return True


# =============================================================================
# Image Timestamp Normalization (for reproducible PCR0)
# =============================================================================

def normalize_docker_image(image_name: str, normalized_name: str) -> bool:
    """
    Normalize a Docker image to have deterministic timestamps.
    
    Docker builds are non-deterministic - layer tar archives contain file 
    modification timestamps from build time. This causes different PCR0 values
    even with identical source code.
    
    Solution: After building, normalize ALL timestamps to epoch 0:
    1. Export image to tar
    2. Rewrite each layer tar with mtime=0 for all files
    3. Update config JSON with created="1970-01-01T00:00:00Z"
    4. Recompute all hashes
    5. Load normalized image
    
    This ensures: Same code → Same normalized image → Same PCR0
    """
    work_dir = Path(tempfile.mkdtemp(prefix="pcr0_normalize_"))
    
    try:
        # Export image
        logger.info(f"[PCR0] Normalizing image {image_name}...")
        export_result = os.system(f"docker save {image_name} -o {work_dir}/orig.tar")
        if export_result != 0:
            logger.error("[PCR0] Failed to export image")
            return False
        
        # Extract
        with tarfile.open(f"{work_dir}/orig.tar", "r") as tar:
            tar.extractall(work_dir)
        
        # Read manifest
        with open(work_dir / "manifest.json") as f:
            manifest = json.load(f)
        
        layers = manifest[0]["Layers"]
        config_path = manifest[0]["Config"]
        
        logger.info(f"[PCR0] Normalizing {len(layers)} layers...")
        
        # Process each layer - normalize tar timestamps
        new_layers = []
        for layer_path in layers:
            full_path = work_dir / layer_path
            norm_path = str(full_path) + ".norm"
            
            # Rewrite tar with all timestamps = 0
            with tarfile.open(str(full_path), "r") as old_tar:
                with tarfile.open(norm_path, "w") as new_tar:
                    for member in old_tar.getmembers():
                        member.mtime = 0  # Epoch timestamp
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
            
            # Remove old layer
            if str(full_path) != str(new_layer_full):
                try:
                    os.remove(full_path)
                except:
                    pass
            
            new_layers.append(new_layer_name)
        
        # Normalize config JSON
        with open(work_dir / config_path) as f:
            config = json.load(f)
        
        config["created"] = "1970-01-01T00:00:00Z"
        
        # Update rootfs layer digests
        new_diff_ids = []
        for layer in new_layers:
            layer_hash = layer.split("/")[-1]
            new_diff_ids.append("sha256:" + layer_hash)
        config["rootfs"]["diff_ids"] = new_diff_ids
        
        # Normalize history timestamps
        if "history" in config:
            for h in config["history"]:
                if "created" in h:
                    h["created"] = "1970-01-01T00:00:00Z"
        
        # Write normalized config
        config_json = json.dumps(config, separators=(",", ":"))
        new_config_hash = hashlib.sha256(config_json.encode()).hexdigest()
        new_config_path = work_dir / "blobs" / "sha256" / new_config_hash
        new_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(new_config_path, "w") as f:
            f.write(config_json)
        
        # Remove old config
        try:
            os.remove(work_dir / config_path)
        except:
            pass
        
        # Update manifest
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
            # Update the digest in index.json to point to new config
            for m in index.get("manifests", []):
                if m.get("digest", "").startswith("sha256:"):
                    m["digest"] = "sha256:" + new_config_hash
            with open(index_path, "w") as f:
                json.dump(index, f)
            logger.info("[PCR0] Updated index.json for OCI format")
        
        # Create normalized tar
        with tarfile.open(f"{work_dir}/normalized.tar", "w") as tar:
            for item in work_dir.iterdir():
                if item.name not in ["orig.tar", "normalized.tar"]:
                    tar.add(item, arcname=item.name)
        
        # Load normalized image
        load_result = os.system(f"docker load -i {work_dir}/normalized.tar 2>/dev/null")
        if load_result != 0:
            logger.error("[PCR0] Failed to load normalized image")
            return False
        
        # Tag the loaded image
        os.system(f"docker tag sha256:{new_config_hash} {normalized_name} 2>/dev/null")
        
        logger.info(f"[PCR0] ✓ Image normalized: {normalized_name}")
        return True
        
    except Exception as e:
        logger.exception(f"[PCR0] Normalization failed: {e}")
        return False
    finally:
        # Cleanup work directory
        try:
            shutil.rmtree(work_dir)
        except:
            pass


# =============================================================================
# Enclave Build
# =============================================================================

# Base image name - built once and cached
BASE_IMAGE_NAME = "validator-base:v1"


async def ensure_base_image_exists(repo_dir: str) -> bool:
    """
    Ensure the base image exists. Build it if not present.
    
    The base image contains yum-installed Python and is NON-DETERMINISTIC.
    However, once built and cached, it's stable. The enclave image built
    on top uses only COPY operations which are deterministic.
    
    This two-stage approach allows reproducible PCR0 values.
    """
    # Check if base image exists
    proc = await asyncio.create_subprocess_exec(
        "docker", "images", "-q", BASE_IMAGE_NAME,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    
    if stdout.decode().strip():
        logger.info(f"[PCR0] Base image {BASE_IMAGE_NAME} already exists")
        return True
    
    # Build base image (this is done ONCE and cached)
    logger.info(f"[PCR0] Building base image {BASE_IMAGE_NAME} (one-time operation)...")
    proc = await asyncio.create_subprocess_exec(
        "sudo", "docker", "build",
        "-f", "validator_tee/Dockerfile.base",
        "-t", BASE_IMAGE_NAME,
        ".",
        cwd=repo_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    
    if proc.returncode != 0:
        logger.error(f"[PCR0] Failed to build base image: {stderr.decode()[-500:]}")
        return False
    
    logger.info(f"[PCR0] ✓ Base image {BASE_IMAGE_NAME} built successfully")
    return True


async def build_enclave_and_extract_pcr0(repo_dir: str) -> Optional[str]:
    """Build the validator enclave and extract PCR0."""
    docker_image = f"validator-enclave-build-{int(time.time())}"
    normalized_image = f"{docker_image}-normalized"
    eif_path = os.path.join(repo_dir, "validator-enclave.eif")
    
    try:
        # Step 0: Ensure base image exists (built once, cached)
        if not await ensure_base_image_exists(repo_dir):
            logger.error("[PCR0] Cannot proceed without base image")
            return None
        
        # Step 1: Build Docker image with --no-cache
        # The base image is cached, so only our code layers rebuild
        logger.info("[PCR0] Building Docker image (--no-cache for code layers)...")
        proc = await asyncio.create_subprocess_exec(
            "sudo", "docker", "build",
            "--no-cache",
            "-f", "validator_tee/Dockerfile.enclave",
            "-t", docker_image,
            ".",
            cwd=repo_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            logger.error(f"[PCR0] Docker build failed: {stderr.decode()[-500:]}")
            return None
        
        # Step 2: NORMALIZE the image for reproducible PCR0
        # This ensures identical PCR0 regardless of when image was built
        logger.info("[PCR0] Normalizing image timestamps for reproducibility...")
        if not normalize_docker_image(docker_image, normalized_image):
            logger.error("[PCR0] Failed to normalize Docker image")
            return None
        
        # Step 3: Build enclave from NORMALIZED image
        logger.info("[PCR0] Building enclave with nitro-cli...")
        proc = await asyncio.create_subprocess_exec(
            "sudo", "nitro-cli", "build-enclave",
            "--docker-uri", normalized_image,
            "--output-file", eif_path,
            cwd=repo_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            logger.error(f"[PCR0] nitro-cli build failed: {stderr.decode()[-500:]}")
            return None
        
        # Parse PCR0 from output
        output = stdout.decode()
        try:
            start = output.find("{")
            end = output.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(output[start:end])
                pcr0 = data.get("Measurements", {}).get("PCR0")
                if pcr0:
                    logger.info(f"[PCR0] Extracted PCR0: {pcr0[:32]}...")
                    return pcr0
        except json.JSONDecodeError as e:
            logger.error(f"[PCR0] Failed to parse nitro-cli output: {e}")
        
        logger.error(f"[PCR0] Could not find PCR0 in output: {output[:500]}")
        return None
        
    finally:
        # Cleanup
        for img in [docker_image, normalized_image]:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "sudo", "docker", "rmi", "-f", img,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
            except Exception:
                pass
        
        try:
            if os.path.exists(eif_path):
                os.remove(eif_path)
        except Exception:
            pass


# =============================================================================
# Content Hash Tracking (for detecting PCR0-relevant changes)
# =============================================================================

def compute_files_content_hash(repo_dir: str) -> Optional[str]:
    """
    Compute a hash of all PCR0-relevant files' contents.
    
    This is used to detect when files actually changed (not just commits).
    Only rebuilds when the content of monitored files changes.
    """
    hasher = hashlib.sha256()
    
    files_found = 0
    for filepath in sorted(MONITORED_FILES):
        full_path = os.path.join(repo_dir, filepath)
        if os.path.exists(full_path):
            try:
                with open(full_path, 'rb') as f:
                    hasher.update(f.read())
                hasher.update(filepath.encode())  # Include path in hash
                files_found += 1
            except Exception as e:
                logger.warning(f"[PCR0] Could not read {filepath}: {e}")
    
    # Also hash files in monitored directories
    for dirpath in sorted(MONITORED_DIRS):
        full_dir = os.path.join(repo_dir, dirpath)
        if os.path.isdir(full_dir):
            for root, dirs, files in os.walk(full_dir):
                for filename in sorted(files):
                    if filename.endswith('.py'):  # Only Python files
                        filepath = os.path.join(root, filename)
                        rel_path = os.path.relpath(filepath, repo_dir)
                        try:
                            with open(filepath, 'rb') as f:
                                hasher.update(f.read())
                            hasher.update(rel_path.encode())
                            files_found += 1
                        except Exception as e:
                            logger.warning(f"[PCR0] Could not read {rel_path}: {e}")
    
    if files_found == 0:
        logger.error("[PCR0] No monitored files found!")
        return None
    
    content_hash = hasher.hexdigest()[:16]  # Short hash for logging
    logger.info(f"[PCR0] Content hash: {content_hash} ({files_found} files)")
    return content_hash


# =============================================================================
# Background Task
# =============================================================================

# Track the last content hash we built for
_last_content_hash: Optional[str] = None

async def check_and_build_pcr0():
    """
    Check for file changes and build PCR0 if needed.
    
    LOGIC:
    1. Fetch latest code from GitHub (sparse checkout - only PCR0 files)
    2. Compute content hash of all monitored files
    3. If content hash changed from last build → rebuild PCR0
    4. Cache the PCR0 (keyed by content hash)
    5. Keep last 3 PCR0 values for validators on different versions
    """
    global _last_content_hash, _build_in_progress, _pcr0_cache
    
    if _build_in_progress:
        logger.info("[PCR0] Build already in progress, skipping")
        return
    
    _build_in_progress = True
    
    try:
        repo_dir = BUILD_DIR
        
        # Clone or update repo (sparse checkout - only PCR0 files)
        print("[PCR0] Fetching latest code from GitHub...")
        logger.info("[PCR0] Fetching latest code from GitHub...")
        if not await clone_or_update_repo(repo_dir):
            logger.error("[PCR0] Failed to update repo")
            return
        
        # Compute content hash of monitored files
        content_hash = compute_files_content_hash(repo_dir)
        if not content_hash:
            logger.error("[PCR0] Failed to compute content hash")
            return
        
        # Check if we already have this content hash cached
        # With pinned Dockerfile, same content = same PCR0, so no need to rebuild
        if content_hash in _pcr0_cache:
            logger.info(f"[PCR0] Content hash {content_hash} already cached, skipping build")
            print(f"[PCR0] Content hash {content_hash} already cached, skipping build")
            _last_content_hash = content_hash
            return
        
        # If we get here, content hash is not in cache - need to build
        print(f"[PCR0] New content detected! Hash: {content_hash} (previous: {_last_content_hash})")
        logger.info(f"[PCR0] New content detected - building PCR0...")
        print(f"[PCR0] Building PCR0 for content hash {content_hash}...")
        
        # Get commit hash for reference (optional, just for logging)
        commits = await get_latest_commits(repo_dir, 1)
        commit_hash = commits[0]["hash"] if commits else "unknown"
        
        # Build PCR0 for current content
        async with _cache_lock:
            logger.info(f"[PCR0] Building enclave for content hash {content_hash} (commit {commit_hash[:8]})...")
            
            pcr0 = await build_enclave_and_extract_pcr0(repo_dir)
            
            if pcr0:
                # Store keyed by CONTENT HASH (not commit hash)
                # This means same code = same key, regardless of commit
                _pcr0_cache[content_hash] = {
                    "pcr0": pcr0,
                    "content_hash": content_hash,
                    "commit_hash": commit_hash,
                    "built_at": datetime.utcnow().isoformat(),
                }
                print(f"[PCR0] ✅ Cached PCR0 for content {content_hash}: {pcr0[:64]}...")
                logger.info(f"[PCR0] ✅ Cached PCR0 for content {content_hash}: {pcr0[:32]}...")
                
                # Prune old entries (keep only last N)
                if len(_pcr0_cache) > PCR0_CACHE_SIZE:
                    # Sort by built_at and keep newest
                    sorted_entries = sorted(
                        _pcr0_cache.items(),
                        key=lambda x: x[1]["built_at"],
                        reverse=True
                    )
                    _pcr0_cache = dict(sorted_entries[:PCR0_CACHE_SIZE])
                    logger.info(f"[PCR0] Pruned cache to {PCR0_CACHE_SIZE} entries")
            else:
                logger.error(f"[PCR0] ❌ Failed to build PCR0 for content {content_hash}")
        
        _last_content_hash = content_hash
        logger.info(f"[PCR0] ✅ Cache updated. Valid PCR0s: {len(_pcr0_cache)}")
        
    except Exception as e:
        logger.exception(f"[PCR0] Error in check_and_build: {e}")
    finally:
        _build_in_progress = False


async def check_prerequisites() -> bool:
    """Check that required tools are available."""
    # Check for nitro-cli
    proc = await asyncio.create_subprocess_exec(
        "which", "nitro-cli",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    
    if proc.returncode != 0:
        logger.error("[PCR0] ❌ nitro-cli not found! PCR0 builder cannot run.")
        return False
    
    logger.info(f"[PCR0] ✓ nitro-cli found: {stdout.decode().strip()}")
    
    # Check for docker
    proc = await asyncio.create_subprocess_exec(
        "which", "docker",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    
    if proc.returncode != 0:
        logger.error("[PCR0] ❌ docker not found! PCR0 builder cannot run.")
        return False
    
    logger.info(f"[PCR0] ✓ docker found: {stdout.decode().strip()}")
    
    # Check for git
    proc = await asyncio.create_subprocess_exec(
        "which", "git",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    
    if proc.returncode != 0:
        logger.error("[PCR0] ❌ git not found! PCR0 builder cannot run.")
        return False
    
    logger.info(f"[PCR0] ✓ git found: {stdout.decode().strip()}")
    
    return True


async def pcr0_builder_task():
    """Background task that runs every 8 minutes."""
    logger.info(f"[PCR0] ========================================")
    logger.info(f"[PCR0] Starting PCR0 builder task")
    logger.info(f"[PCR0] Interval: {PCR0_CHECK_INTERVAL}s ({PCR0_CHECK_INTERVAL // 60} minutes)")
    logger.info(f"[PCR0] Cache size: {PCR0_CACHE_SIZE}")
    logger.info(f"[PCR0] GitHub repo: {GITHUB_REPO_URL}")
    logger.info(f"[PCR0] GitHub branch: {GITHUB_BRANCH}")
    logger.info(f"[PCR0] Build dir: {BUILD_DIR}")
    logger.info(f"[PCR0] ========================================")
    
    # Check prerequisites
    if not await check_prerequisites():
        logger.error("[PCR0] Prerequisites check failed. PCR0 builder disabled.")
        return
    
    # Initial build on startup
    logger.info("[PCR0] Running initial PCR0 build...")
    await check_and_build_pcr0()
    
    logger.info(f"[PCR0] Initial build complete. Cache status: {get_cache_status()}")
    
    while True:
        await asyncio.sleep(PCR0_CHECK_INTERVAL)
        logger.info(f"[PCR0] Timer fired - checking for updates...")
        await check_and_build_pcr0()


def start_pcr0_builder():
    """Start the background PCR0 builder task."""
    asyncio.create_task(pcr0_builder_task())
    logger.info("[PCR0] Background builder task started")
    print("🔐 [PCR0] Background builder task started")
    print(f"   Repo: {GITHUB_REPO_URL}")
    print(f"   Branch: {GITHUB_BRANCH}")
    print(f"   Interval: {PCR0_CHECK_INTERVAL // 60} minutes")
    print(f"   Cache size: {PCR0_CACHE_SIZE} versions")
    print(f"   Mode: PINNED DOCKERFILE (reproducible builds)")
    print(f"   TTL: None (pinned builds are deterministic)")


# =============================================================================
# API for verification
# =============================================================================

def verify_pcr0(pcr0: str) -> Dict:
    """
    Verify a PCR0 value against our computed cache.
    
    The cache stores PCR0 values keyed by CONTENT HASH of monitored files.
    This means:
    - Same code = same PCR0 (regardless of how many commits)
    - Only 3 different code versions are cached
    - Validators on older code versions are still accepted
    
    Returns:
        {
            "valid": bool,
            "commit_hash": str or None,
            "content_hash": str or None,
            "message": str,
            "cache_size": int,
        }
    """
    for content_hash, entry in _pcr0_cache.items():
        if entry["pcr0"] == pcr0:
            return {
                "valid": True,
                "commit_hash": entry.get("commit_hash", "unknown"),
                "content_hash": content_hash,
                "built_at": entry.get("built_at"),
                "message": f"PCR0 matches content {content_hash} (commit {entry.get('commit_hash', 'unknown')[:8]})",
                "cache_size": len(_pcr0_cache),
            }
    
    return {
        "valid": False,
        "commit_hash": None,
        "content_hash": None,
        "message": f"PCR0 not in cache. Valid PCR0s: {len(_pcr0_cache)}",
        "cache_size": len(_pcr0_cache),
        "cached_pcr0s": [e["pcr0"][:32] + "..." for e in _pcr0_cache.values()],
    }

