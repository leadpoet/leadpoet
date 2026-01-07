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

# How many commits to keep PCR0 for
PCR0_CACHE_SIZE = int(os.environ.get("PCR0_CACHE_SIZE", "3"))

# Files that affect PCR0 (if any of these change, rebuild)
MONITORED_FILES: Set[str] = {
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
    sparse_paths = [
        "validator_tee/",
        "leadpoet_canonical/",
        "neurons/validator.py",
        "validator_models/automated_checks.py",
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
        proc = await asyncio.create_subprocess_exec(
            "git", "sparse-checkout", "set", *sparse_paths,
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
# Enclave Build
# =============================================================================

async def build_enclave_and_extract_pcr0(repo_dir: str) -> Optional[str]:
    """Build the validator enclave and extract PCR0."""
    docker_image = f"validator-enclave-build-{int(time.time())}"
    eif_path = os.path.join(repo_dir, "validator-enclave.eif")
    
    try:
        # Step 1: Build Docker image
        logger.info("[PCR0] Building Docker image...")
        proc = await asyncio.create_subprocess_exec(
            "docker", "build",
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
        
        # Step 2: Build enclave and extract PCR0
        logger.info("[PCR0] Building enclave with nitro-cli...")
        proc = await asyncio.create_subprocess_exec(
            "sudo", "nitro-cli", "build-enclave",
            "--docker-uri", docker_image,
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
        # Output format: {"Measurements": {"PCR0": "...", ...}}
        output = stdout.decode()
        try:
            # Find JSON in output
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
        try:
            # Remove Docker image
            proc = await asyncio.create_subprocess_exec(
                "docker", "rmi", "-f", docker_image,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
        except Exception:
            pass
        
        try:
            # Remove EIF file
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
        if content_hash in _pcr0_cache:
            logger.info(f"[PCR0] Content hash {content_hash} already cached, skipping build")
            _last_content_hash = content_hash
            return
        
        # Check if content actually changed
        if _last_content_hash == content_hash:
            logger.info(f"[PCR0] No changes to monitored files (hash: {content_hash})")
            return
        
        logger.info(f"[PCR0] Content changed! Old: {_last_content_hash}, New: {content_hash}")
        logger.info(f"[PCR0] Building PCR0 for content hash {content_hash}...")
        
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


async def pcr0_builder_task():
    """Background task that runs every 8 minutes."""
    logger.info(f"[PCR0] Starting PCR0 builder task (interval: {PCR0_CHECK_INTERVAL}s)")
    
    # Initial build on startup
    await check_and_build_pcr0()
    
    while True:
        await asyncio.sleep(PCR0_CHECK_INTERVAL)
        await check_and_build_pcr0()


def start_pcr0_builder():
    """Start the background PCR0 builder task."""
    asyncio.create_task(pcr0_builder_task())
    logger.info("[PCR0] Background builder task started")


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

