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

# {commit_hash: {"pcr0": "...", "built_at": timestamp, "files_hash": "..."}}
_pcr0_cache: Dict[str, Dict] = {}
_cache_lock = asyncio.Lock()

# Last known commit hash (to detect changes)
_last_commit: Optional[str] = None

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
        "cached_commits": list(_pcr0_cache.keys()),
        "cached_pcr0s": get_cached_pcr0_values(),
        "last_commit": _last_commit,
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


async def get_changed_files(repo_dir: str, commit1: str, commit2: str) -> Set[str]:
    """Get files changed between two commits."""
    proc = await asyncio.create_subprocess_exec(
        "git", "diff", "--name-only", commit1, commit2,
        cwd=repo_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    
    if proc.returncode != 0:
        logger.error(f"[PCR0] git diff failed: {stderr.decode()}")
        return set()
    
    return set(stdout.decode().strip().split("\n"))


def should_rebuild(changed_files: Set[str]) -> bool:
    """Check if any changed file affects PCR0."""
    for f in changed_files:
        # Check exact file matches
        if f in MONITORED_FILES:
            return True
        # Check directory matches
        for d in MONITORED_DIRS:
            if f.startswith(d):
                return True
    return False


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


async def checkout_commit(repo_dir: str, commit_hash: str) -> bool:
    """Checkout a specific commit."""
    proc = await asyncio.create_subprocess_exec(
        "git", "checkout", commit_hash,
        cwd=repo_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    
    if proc.returncode != 0:
        logger.error(f"[PCR0] git checkout {commit_hash} failed: {stderr.decode()}")
        return False
    
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
# Background Task
# =============================================================================

async def check_and_build_pcr0():
    """Check for new commits and build PCR0 if needed."""
    global _last_commit, _build_in_progress, _pcr0_cache
    
    if _build_in_progress:
        logger.info("[PCR0] Build already in progress, skipping")
        return
    
    _build_in_progress = True
    
    try:
        repo_dir = BUILD_DIR
        
        # Clone or update repo
        logger.info("[PCR0] Updating repo from GitHub...")
        if not await clone_or_update_repo(repo_dir):
            logger.error("[PCR0] Failed to update repo")
            return
        
        # Get latest commits
        commits = await get_latest_commits(repo_dir, PCR0_CACHE_SIZE)
        if not commits:
            logger.error("[PCR0] No commits found")
            return
        
        latest_commit = commits[0]["hash"]
        
        # Check if we need to rebuild
        need_rebuild = False
        if _last_commit is None:
            # First run, build all
            need_rebuild = True
            logger.info("[PCR0] First run, building all commits")
        elif _last_commit != latest_commit:
            # New commit, check if relevant files changed
            changed = await get_changed_files(repo_dir, _last_commit, latest_commit)
            if should_rebuild(changed):
                need_rebuild = True
                logger.info(f"[PCR0] Monitored files changed: {changed & (MONITORED_FILES | set(f for f in changed for d in MONITORED_DIRS if f.startswith(d)))}")
            else:
                logger.info(f"[PCR0] New commit but no monitored files changed")
                _last_commit = latest_commit
        else:
            logger.info("[PCR0] No new commits")
        
        if not need_rebuild:
            return
        
        # Build PCR0 for each commit we don't have cached
        async with _cache_lock:
            for commit in commits:
                commit_hash = commit["hash"]
                
                # Skip if already cached
                if commit_hash in _pcr0_cache:
                    logger.info(f"[PCR0] Commit {commit_hash[:8]} already cached")
                    continue
                
                # Checkout and build
                logger.info(f"[PCR0] Building PCR0 for commit {commit_hash[:8]}...")
                
                if not await checkout_commit(repo_dir, commit_hash):
                    continue
                
                pcr0 = await build_enclave_and_extract_pcr0(repo_dir)
                
                if pcr0:
                    _pcr0_cache[commit_hash] = {
                        "pcr0": pcr0,
                        "built_at": datetime.utcnow().isoformat(),
                        "commit_message": commit["message"],
                    }
                    logger.info(f"[PCR0] ✅ Cached PCR0 for {commit_hash[:8]}: {pcr0[:32]}...")
                else:
                    logger.error(f"[PCR0] ❌ Failed to build PCR0 for {commit_hash[:8]}")
            
            # Prune old entries (keep only last N)
            if len(_pcr0_cache) > PCR0_CACHE_SIZE:
                # Sort by built_at and keep newest
                sorted_commits = sorted(
                    _pcr0_cache.items(),
                    key=lambda x: x[1]["built_at"],
                    reverse=True
                )
                _pcr0_cache = dict(sorted_commits[:PCR0_CACHE_SIZE])
                logger.info(f"[PCR0] Pruned cache to {PCR0_CACHE_SIZE} entries")
        
        _last_commit = latest_commit
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
    
    Returns:
        {
            "valid": bool,
            "commit": str or None,
            "message": str,
            "cache_size": int,
        }
    """
    for commit_hash, entry in _pcr0_cache.items():
        if entry["pcr0"] == pcr0:
            return {
                "valid": True,
                "commit": commit_hash,
                "commit_message": entry.get("commit_message", ""),
                "built_at": entry.get("built_at"),
                "message": f"PCR0 matches commit {commit_hash[:8]}",
                "cache_size": len(_pcr0_cache),
            }
    
    return {
        "valid": False,
        "commit": None,
        "message": f"PCR0 not in cache. Valid PCR0s: {len(_pcr0_cache)}",
        "cache_size": len(_pcr0_cache),
        "cached_pcr0s": [e["pcr0"][:32] + "..." for e in _pcr0_cache.values()],
    }

