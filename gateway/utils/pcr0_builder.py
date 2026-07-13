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
- leadpoet_verifier/*
- research_lab/*
- gateway/research_lab/*
- gateway qualification/db/task helpers used by Research Lab
- qualification/scoring/*
- Research Lab worker entrypoint scripts
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
PCR0_CACHE_SIZE = int(os.environ.get("PCR0_CACHE_SIZE", "20"))
VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT = os.path.expanduser(
    os.environ.get(
        "VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT",
        "~/.cache/leadpoet-v2-artifacts/validator-runtime",
    )
)

# Git history depth for the builder's working clone.  The historical warm-up
# advertises "last N commits", but with a depth-1 clone the repo only ever
# contains HEAD, so after any gateway restart the (in-memory) PCR0 cache is
# rebuilt with HEAD's PCR0 alone — a validator whose enclave was built even
# one monitored commit earlier stops verifying dynamically and every weight
# submission 403s until its measurement is hand-added to the static
# allowlist.  History is metadata-only under --filter=blob:none (blobs fetch
# on demand at checkout), so a deeper clone costs almost nothing.
PCR0_GIT_HISTORY_DEPTH = max(1, int(os.environ.get("PCR0_GIT_HISTORY_DEPTH", "40")))

# Give the gateway process time to finish serving health/readiness before the
# Docker/Nitro cache warmer starts competing for host resources. Static PCR0
# allowlist acceptance remains available while this delayed warmup runs.
PCR0_STARTUP_WARM_DELAY_SECONDS = int(os.environ.get("PCR0_STARTUP_WARM_DELAY_SECONDS", "60"))
PCR0_STARTUP_HISTORICAL_WARM_ENABLED = os.environ.get(
    "PCR0_STARTUP_HISTORICAL_WARM_ENABLED",
    "true",
).lower() in {"1", "true", "yes", "y", "on"}

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
    "validator_tee/runtime-artifacts-v2.lock.json",
    "validator_tee/scripts/stage_runtime_artifacts_v2.py",
    "validator_tee/enclave/__init__.py",
    "validator_tee/enclave/nsm_lib.py",
    "validator_tee/enclave/tee_service.py",
    "neurons/validator.py",
    "validator_models/automated_checks.py",
    "gateway/__init__.py",
    "gateway/qualification/__init__.py",
    "gateway/qualification/models.py",
    "gateway/qualification/config.py",
    "gateway/db/__init__.py",
    "gateway/db/client.py",
    "gateway/tasks/__init__.py",
    "gateway/tasks/icp_generator.py",
    "qualification/__init__.py",
    "scripts/run_research_lab_hosted_worker.py",
    "scripts/run_research_lab_hosted_worker_fleet.py",
    "scripts/run_research_lab_scoring_worker.py",
    "scripts/run_research_lab_scoring_worker_fleet.py",
}

# Directories where any file change triggers rebuild
MONITORED_DIRS: Set[str] = {
    "leadpoet_canonical/",
    "leadpoet_verifier/",
    "research_lab/",
    "gateway/research_lab/",
    "gateway/qualification/utils/",
    "qualification/scoring/",
}

MONITORED_DIR_SUFFIXES: Set[str] = {".py", ".json", ".txt"}

# Working directory for builds
BUILD_DIR = os.environ.get("PCR0_BUILD_DIR", "/tmp/pcr0_builder")

# Paths copied by validator_tee/Dockerfile.enclave. These are the exact local
# build inputs that must be clean and mode-normalized before Docker sees them.
PCR0_COPY_PATHS: List[str] = [
    "validator_tee/enclave",
    "validator_tee/runtime-artifacts-v2.lock.json",
    "leadpoet_canonical",
    "leadpoet_verifier",
    "research_lab",
    "gateway/research_lab",
    "gateway/qualification/utils",
    "qualification/scoring",
    "gateway/__init__.py",
    "gateway/qualification/__init__.py",
    "gateway/qualification/models.py",
    "gateway/qualification/config.py",
    "gateway/db/__init__.py",
    "gateway/db/client.py",
    "gateway/tasks/__init__.py",
    "gateway/tasks/icp_generator.py",
    "qualification/__init__.py",
    "scripts/run_research_lab_hosted_worker.py",
    "scripts/run_research_lab_hosted_worker_fleet.py",
    "scripts/run_research_lab_scoring_worker.py",
    "scripts/run_research_lab_scoring_worker_fleet.py",
    "neurons/validator.py",
    "validator_models/automated_checks.py",
]


def validator_v2_artifacts_required(repo_dir: str) -> bool:
    """Return whether the checked-out validator Dockerfile uses V2 artifacts."""
    dockerfile_path = os.path.join(repo_dir, "validator_tee", "Dockerfile.enclave")
    with open(dockerfile_path, "r", encoding="utf-8") as handle:
        return ".validator-tee-artifacts/" in handle.read()


# =============================================================================
# Cache
# =============================================================================

# Cache structure (keyed by source content plus base-image identity, not commit hash):
# {cache_key: {"pcr0": "...", "content_hash": "...", "base_image_stamp": "...", "commit_hash": "...", "commit_timestamp": "...", "built_at": timestamp}}
# This means: same code content + same base image = same cache key, regardless of commits
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
        "cached_cache_keys": list(_pcr0_cache.keys()),
        "cached_content_hashes": [v.get("content_hash", k) for k, v in _pcr0_cache.items()],
        "cached_pcr0s": get_cached_pcr0_values(),
        "cache_entries": [
            {
                "cache_key": k,
                "content_hash": v.get("content_hash", "?"),
                "base_image_stamp": v.get("base_image_stamp", "?"),
                "commit_hash": v.get("commit_hash", "?")[:8],
                "commit_timestamp": v.get("commit_timestamp"),
                "pcr0": v["pcr0"][:32] + "...",
                "built_at": v.get("built_at"),
            }
            for k, v in _pcr0_cache.items()
        ],
        "build_in_progress": _build_in_progress,
        "cache_size": len(_pcr0_cache),
        "startup_warm_delay_seconds": PCR0_STARTUP_WARM_DELAY_SECONDS,
        "startup_historical_warm_enabled": PCR0_STARTUP_HISTORICAL_WARM_ENABLED,
    }


# =============================================================================
# Git Operations
# =============================================================================

def _cache_commit_timestamp(entry: Dict) -> int:
    try:
        return int(entry.get("commit_timestamp") or -1)
    except (TypeError, ValueError):
        return -1


def _prune_pcr0_cache() -> None:
    """Prune cache by commit recency, with build time only as a fallback tie-breaker."""
    global _pcr0_cache

    if len(_pcr0_cache) <= PCR0_CACHE_SIZE:
        return

    sorted_entries = sorted(
        _pcr0_cache.items(),
        key=lambda item: (
            _cache_commit_timestamp(item[1]),
            item[1].get("built_at") or "",
        ),
        reverse=True,
    )
    removed_entries = sorted_entries[PCR0_CACHE_SIZE:]
    _pcr0_cache = dict(sorted_entries[:PCR0_CACHE_SIZE])
    removed = ", ".join(
        f"{entry.get('commit_hash', 'unknown')[:8]}:{entry.get('pcr0', '')[:16]}"
        for _key, entry in removed_entries
    )
    logger.info(f"[PCR0] Pruned cache to {PCR0_CACHE_SIZE} entries by commit recency; removed {removed}")


async def get_latest_commits(repo_dir: str, count: int = 3) -> List[Dict]:
    """Get the latest N commits from the repo."""
    proc = await asyncio.create_subprocess_exec(
        "git", "log", f"-{count}", "--format=%H|%ct|%s|%ai",
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
            parts = line.split("|", 3)
            commits.append({
                "hash": parts[0],
                "timestamp": parts[1] if len(parts) > 1 else "",
                "message": parts[2] if len(parts) > 2 else "",
                "date": parts[3] if len(parts) > 3 else "",
            })
    
    return commits


async def clone_or_update_repo(repo_dir: str) -> bool:
    """
    Clone or update the repo using sparse checkout.
    
    OPTIMIZATION: Only fetches the files needed for PCR0 verification:
    - validator_tee/ (Dockerfile and enclave code)
    - leadpoet_canonical/ (canonical modules)
    - leadpoet_verifier/ (open verifier and golden vectors)
    - research_lab/ and gateway/research_lab/ (Research Lab execution/audit logic)
    - qualification scoring modules used by Research Lab scoring
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
        "/leadpoet_verifier/",
        "/research_lab/",
        "/gateway/__init__.py",
        "/gateway/research_lab/",
        "/gateway/qualification/__init__.py",
        "/gateway/qualification/models.py",
        "/gateway/qualification/config.py",
        "/gateway/qualification/utils/",
        "/gateway/db/__init__.py",
        "/gateway/db/client.py",
        "/gateway/tasks/__init__.py",
        "/gateway/tasks/icp_generator.py",
        "/qualification/__init__.py",
        "/qualification/scoring/",
        "/neurons/validator.py",
        "/validator_models/automated_checks.py",
        "/scripts/run_research_lab_hosted_worker.py",
        "/scripts/run_research_lab_hosted_worker_fleet.py",
        "/scripts/run_research_lab_scoring_worker.py",
        "/scripts/run_research_lab_scoring_worker_fleet.py",
    ]
    
    if os.path.exists(os.path.join(repo_dir, ".git")):
        # Update existing repo - fetch, reset, AND re-apply sparse-checkout.
        #
        # BUG FIX (2026-04-23): The original implementation only ran
        # `sparse-checkout set` on the fresh-clone branch.  We observed a
        # production state where the repo's sparse-checkout file was stuck
        # at the default cone-mode patterns (``/*`` + ``!/*/`` — i.e. only
        # root-level files, no directories) even though the fresh-clone
        # path had long since run.  With that state, `validator_tee/`,
        # `leadpoet_canonical/`, and `neurons/` are all absent from the
        # working tree, so `compute_dockerfile_base_hash()` fails with
        # "Cannot read Dockerfile.base: No such file or directory" and
        # the whole PCR0 pipeline never produces a new cache entry —
        # every weight submission gets rejected with 403 "PCR0 not
        # recognized".  Root cause is unclear (possibly a git version
        # behavior change, a previous crash between `clone --sparse` and
        # `sparse-checkout set`, or a stray `git sparse-checkout init`
        # somewhere), but the fix is trivially safe: re-apply the patterns
        # on every update so the state is self-healing.
        proc = await asyncio.create_subprocess_exec(
            "git", "fetch", "--depth", str(PCR0_GIT_HISTORY_DEPTH), "origin", GITHUB_BRANCH,
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

        # Self-heal the sparse-checkout config.  `set --no-cone` is
        # idempotent when the patterns already match, and re-materializes
        # the working tree if they don't.  If this fails, the broken
        # state from a stale clone would persist forever, so on failure
        # we nuke the repo and re-clone from scratch.
        proc = await asyncio.create_subprocess_exec(
            "git", "sparse-checkout", "set", "--no-cone", *sparse_paths,
            cwd=repo_dir,
            env=git_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning(
                f"[PCR0] sparse-checkout re-apply failed on update branch: "
                f"{stderr.decode()[:300]} — nuking repo and re-cloning"
            )
            shutil.rmtree(repo_dir)
            return await clone_or_update_repo(repo_dir)

        # Verify critical monitored files actually materialized.  If the
        # working tree still doesn't have `validator_tee/Dockerfile.base`
        # after the above, something else is wrong and a fresh clone is
        # the only recovery.
        dockerfile_base = os.path.join(repo_dir, "validator_tee", "Dockerfile.base")
        if not os.path.exists(dockerfile_base):
            logger.warning(
                f"[PCR0] {dockerfile_base} still missing after sparse-checkout "
                f"re-apply — working tree is corrupt, re-cloning"
            )
            shutil.rmtree(repo_dir)
            return await clone_or_update_repo(repo_dir)

        logger.info("[PCR0] Repo updated via fetch (sparse-checkout re-applied)")
    else:
        # Fresh clone with sparse checkout (minimal download)
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        os.makedirs(repo_dir, exist_ok=True)
        
        logger.info(f"[PCR0] Sparse cloning {GITHUB_REPO_URL}...")
        
        # Step 1: Clone with sparse checkout enabled (downloads only .git metadata)
        proc = await asyncio.create_subprocess_exec(
            "git", "clone",
            "--depth", str(PCR0_GIT_HISTORY_DEPTH),  # History for warm-up's last-N-commits build
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
        
        # Process each layer - normalize tar timestamps AND file order
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
            
            normalized_layer_paths[layer_path] = new_layer_name
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
# External stamp used to track Dockerfile.base content hash without mutating
# Docker image metadata. Gateway and validator base images must have identical
# Docker config, so do not use labels for freshness tracking.
BASE_IMAGE_STAMP_FILENAME = ".validator-base.dockerfile.sha256"


def compute_dockerfile_base_hash(repo_dir: str) -> Optional[str]:
    """Compute SHA256 hash of Dockerfile.base content."""
    dockerfile_path = os.path.join(repo_dir, "validator_tee", "Dockerfile.base")
    try:
        with open(dockerfile_path, 'rb') as f:
            content = f.read()
        return hashlib.sha256(content).hexdigest()[:16]
    except Exception as e:
        logger.error(f"[PCR0] Cannot read Dockerfile.base: {e}")
        return None


def get_base_image_stamp_path(repo_dir: str) -> str:
    """Return path to external base-image freshness stamp."""
    return os.path.join(repo_dir, BASE_IMAGE_STAMP_FILENAME)


def read_base_image_stamp(repo_dir: str) -> tuple[Optional[str], Optional[str]]:
    """Read the Dockerfile.base content hash and image id stamp, if present."""
    try:
        with open(get_base_image_stamp_path(repo_dir), "r", encoding="utf-8") as f:
            value = f.read().strip()
        if not value:
            return None, None
        parts = value.split(maxsplit=1)
        if len(parts) == 1:
            return parts[0], None
        return parts[0], parts[1]
    except FileNotFoundError:
        return None, None
    except Exception as e:
        logger.warning(f"[PCR0] Could not read base image stamp: {e}")
        return None, None


def write_base_image_stamp(repo_dir: str, dockerfile_hash: str, image_id: str) -> None:
    """Write the Dockerfile.base content hash and local image id stamp."""
    try:
        with open(get_base_image_stamp_path(repo_dir), "w", encoding="utf-8") as f:
            f.write(f"{dockerfile_hash} {image_id}\n")
    except Exception as e:
        logger.warning(f"[PCR0] Could not write base image stamp: {e}")


async def inspect_base_image_id() -> Optional[str]:
    """Return the local Docker image id for the validator base image, if present."""
    proc = await asyncio.create_subprocess_exec(
        "docker", "image", "inspect", "-f", "{{.Id}}", BASE_IMAGE_NAME,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _stderr = await proc.communicate()
    if proc.returncode != 0:
        return None
    return stdout.decode().strip() or None


def format_base_image_stamp(repo_dir: str) -> str:
    dockerfile_hash, image_id = read_base_image_stamp(repo_dir)
    if not dockerfile_hash:
        return "missing"
    return f"{dockerfile_hash}:{image_id or 'legacy-no-image-id'}"


def build_pcr0_cache_key(content_hash: str, repo_dir: str) -> str:
    """Include the base-image identity in the PCR0 cache key."""
    return f"{content_hash}:{format_base_image_stamp(repo_dir)}"


async def ensure_base_image_exists(repo_dir: str) -> bool:
    """
    Ensure the base image exists AND is up-to-date with current Dockerfile.base.
    
    The base image contains yum-installed Python and is NON-DETERMINISTIC.
    However, once built and cached, it's stable. The enclave image built
    on top uses only COPY operations which are deterministic.
    
    This function tracks Dockerfile.base content via an external stamp file.
    The stamp avoids gateway-only Docker labels, which change image metadata
    and can break PCR0 reproducibility versus the validator build.
    """
    # Compute current Dockerfile.base hash
    current_hash = compute_dockerfile_base_hash(repo_dir)
    if not current_hash:
        logger.error("[PCR0] Cannot compute Dockerfile.base hash")
        return False
    
    current_image_id = await inspect_base_image_id()
    image_exists = bool(current_image_id)
    
    if image_exists:
        # Check if the existing image matches current Dockerfile.base.
        # The hash and image ID are tracked outside Docker image config to keep
        # gateway and validator base builds byte/config equivalent while still
        # detecting stale same-Dockerfile rebuilds.
        existing_hash, existing_image_id = read_base_image_stamp(repo_dir)
        
        if existing_hash == current_hash and existing_image_id == current_image_id:
            logger.info(
                f"[PCR0] Base image {BASE_IMAGE_NAME} up-to-date "
                f"(hash: {current_hash}, image: {str(current_image_id)[:32]})"
            )
            return True
        
        # Hash, old one-field stamp, or image mismatch - rebuild.
        logger.info(
            "[PCR0] Base image stamp stale. "
            f"old_hash={existing_hash} new_hash={current_hash} "
            f"stamped_image={existing_image_id} current_image={current_image_id}"
        )
        logger.info(f"[PCR0] Deleting stale base image {BASE_IMAGE_NAME}...")
        
        # Delete the old base image
        proc = await asyncio.create_subprocess_exec(
            "docker", "rmi", "-f", BASE_IMAGE_NAME,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        # Ignore errors - image might be in use, but we'll build a new one anyway
    
    # Build base image with the same command shape used by the validator script.
    # --no-cache is intentional: the base layer contains yum/rpm work and must
    # be rebuilt from the pinned Dockerfile inputs when the stamp is absent or
    # stale, not silently resurrected from old BuildKit cache.
    logger.info(f"[PCR0] Building base image {BASE_IMAGE_NAME} (hash: {current_hash})...")
    proc = await asyncio.create_subprocess_exec(
        "sudo", "docker", "build",
        "--no-cache",
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
    
    built_image_id = await inspect_base_image_id()
    if not built_image_id:
        logger.error("[PCR0] Base image built but image ID could not be inspected")
        return False

    write_base_image_stamp(repo_dir, current_hash, built_image_id)
    logger.info(
        f"[PCR0] ✓ Base image {BASE_IMAGE_NAME} built successfully "
        f"(hash: {current_hash}, image: {str(built_image_id)[:32]})"
    )
    return True


async def build_enclave_and_extract_pcr0(repo_dir: str) -> Optional[str]:
    """Build the validator enclave and extract PCR0."""
    docker_image = f"validator-enclave-build-{int(time.time())}"
    normalized_image = f"{docker_image}-normalized:latest"
    eif_path = os.path.join(repo_dir, "validator-enclave.eif")
    
    try:
        # Step 0: Ensure base image exists (built once, cached)
        if not await ensure_base_image_exists(repo_dir):
            logger.error("[PCR0] Cannot proceed without base image")
            return None
        
        # Step 0.4: Clean and normalize the exact Docker COPY inputs.
        #
        # The gateway uses a sparse GitHub checkout, but this keeps the builder
        # self-healing if a reused /tmp/pcr0_builder ever accumulates untracked
        # files in paths copied into the enclave image.
        logger.info("[PCR0] Cleaning PCR0 Docker context...")
        proc = await asyncio.create_subprocess_exec(
            "git", "clean", "-ffdx", "--", *PCR0_COPY_PATHS,
            cwd=repo_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        artifact_dir = os.path.join(repo_dir, ".validator-tee-artifacts")
        try:
            requires_v2_artifacts = validator_v2_artifacts_required(repo_dir)
        except OSError as exc:
            logger.error("[PCR0] Cannot inspect validator enclave Dockerfile: %s", exc)
            return None

        # The historical cache intentionally rebuilds commits from before the
        # V2 runtime-artifact contract existed. Those Dockerfiles neither copy
        # nor need `.validator-tee-artifacts`; requiring the newer staging
        # script while an old commit is checked out makes every legacy PCR0
        # rebuild fail before Docker starts.
        if requires_v2_artifacts:
            artifact_script = os.path.join(
                repo_dir, "validator_tee", "scripts", "stage_runtime_artifacts_v2.py"
            )
            artifact_lock = os.path.join(
                repo_dir, "validator_tee", "runtime-artifacts-v2.lock.json"
            )
            if not os.path.isfile(artifact_script) or not os.path.isfile(artifact_lock):
                logger.error(
                    "[PCR0] Validator Dockerfile requires V2 artifacts but its "
                    "staging contract is incomplete"
                )
                return None
            proc = await asyncio.create_subprocess_exec(
                "python3",
                artifact_script,
                "--lock",
                artifact_lock,
                "--output-dir",
                artifact_dir,
                "--offline-artifact-root",
                VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT,
                cwd=repo_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _artifact_stdout, artifact_stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.error(
                    "[PCR0] Failed to stage validator V2 artifacts: %s",
                    artifact_stderr.decode()[-500:],
                )
                return None
        else:
            shutil.rmtree(artifact_dir, ignore_errors=True)
            logger.info("[PCR0] Legacy validator commit does not require V2 artifacts")

        # __pycache__ directories and .pyc files can differ between machines.
        for rel_path in PCR0_COPY_PATHS:
            full_copy_path = os.path.join(repo_dir, rel_path)
            if os.path.isdir(full_copy_path):
                for root, dirs, files in os.walk(full_copy_path):
                    for d in list(dirs):
                        if d == "__pycache__":
                            shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                    for f in files:
                        if f.endswith(".pyc") or f.endswith(".pyo"):
                            try:
                                os.remove(os.path.join(root, f))
                            except:
                                pass
        
        # Step 0.5: Normalize file permissions for reproducibility
        # Docker COPY includes file permissions in layer hash
        # Different machines may have different umask settings (644 vs 664)
        # We normalize every copied file to 644 and every copied directory to
        # 755 to ensure identical layer hashes.
        logger.info("[PCR0] Normalizing PCR0 Docker context permissions...")
        copied_files = []
        for rel_path in PCR0_COPY_PATHS:
            full_path = os.path.join(repo_dir, rel_path)
            if os.path.isdir(full_path):
                for root, dirs, files in os.walk(full_path):
                    for dirname in dirs:
                        os.chmod(os.path.join(root, dirname), 0o755)
                    os.chmod(root, 0o755)
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        if os.path.isfile(fpath):
                            os.chmod(fpath, 0o644)
                            copied_files.append(os.path.relpath(fpath, repo_dir))
            elif os.path.isfile(full_path):
                os.chmod(full_path, 0o644)
                copied_files.append(rel_path)

        # Legacy commits deliberately build without `.validator-tee-artifacts`
        # (their Dockerfile never copies it), so only V2 commits may require
        # the staged directory here.
        if requires_v2_artifacts:
            if not os.path.isdir(artifact_dir):
                logger.error("[PCR0] Validator V2 artifact directory is missing")
                return None
            for root, dirs, files in os.walk(artifact_dir):
                for dirname in dirs:
                    directory = os.path.join(root, dirname)
                    os.chmod(directory, 0o755)
                    os.utime(directory, (0, 0))
                os.chmod(root, 0o755)
                os.utime(root, (0, 0))
                for fname in files:
                    fpath = os.path.join(root, fname)
                    os.chmod(fpath, 0o644)
                    os.utime(fpath, (0, 0))
                    copied_files.append(os.path.relpath(fpath, repo_dir))

        manifest_hash = hashlib.sha256()
        for rel_path in sorted(set(copied_files)):
            full_path = os.path.join(repo_dir, rel_path)
            try:
                st = os.stat(full_path)
                manifest_hash.update(f"{st.st_mode & 0o777} {st.st_size} {rel_path}\n".encode())
                with open(full_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        manifest_hash.update(chunk)
                manifest_hash.update(b"\n")
            except FileNotFoundError:
                manifest_hash.update(f"MISSING {rel_path}\n".encode())
        logger.info(f"[PCR0] PCR0 Docker context manifest_sha256={manifest_hash.hexdigest()}")
        
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
                    if any(filename.endswith(suffix) for suffix in MONITORED_DIR_SUFFIXES):
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
    5. Keep recent PCR0 values for validators on different versions
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

        if not await ensure_base_image_exists(repo_dir):
            logger.error("[PCR0] Failed to prepare base image")
            return

        cache_key = build_pcr0_cache_key(content_hash, repo_dir)
        
        # Check if we already have this content hash cached
        # With pinned Dockerfile, same content = same PCR0, so no need to rebuild
        if cache_key in _pcr0_cache:
            logger.info(f"[PCR0] Cache key {cache_key} already cached, skipping build")
            print(f"[PCR0] Cache key {cache_key} already cached, skipping build")
            _last_content_hash = content_hash
            return
        
        # If we get here, content hash is not in cache - need to build
        print(f"[PCR0] New content detected! Hash: {content_hash} (previous: {_last_content_hash})")
        logger.info(f"[PCR0] New content detected - building PCR0...")
        print(f"[PCR0] Building PCR0 for content hash {content_hash}...")
        
        # Get commit hash for reference (optional, just for logging)
        commits = await get_latest_commits(repo_dir, 1)
        commit_hash = commits[0]["hash"] if commits else "unknown"
        commit_timestamp = commits[0].get("timestamp", "") if commits else ""
        
        # Build PCR0 for current content
        async with _cache_lock:
            logger.info(f"[PCR0] Building enclave for content hash {content_hash} (commit {commit_hash[:8]})...")
            
            pcr0 = await build_enclave_and_extract_pcr0(repo_dir)
            
            if pcr0:
                # Store keyed by source content plus base image identity. The
                # Dockerfile.base hash alone is not enough because the base
                # layer can be rebuilt to a different image ID from the same
                # source; that changes PCR0.
                _pcr0_cache[cache_key] = {
                    "pcr0": pcr0,
                    "content_hash": content_hash,
                    "cache_key": cache_key,
                    "base_image_stamp": format_base_image_stamp(repo_dir),
                    "commit_hash": commit_hash,
                    "commit_timestamp": commit_timestamp,
                    "built_at": datetime.utcnow().isoformat(),
                }
                print(f"[PCR0] ✅ Cached PCR0 for key {cache_key}: {pcr0[:64]}...")
                logger.info(f"[PCR0] ✅ Cached PCR0 for key {cache_key}: {pcr0[:32]}...")
                
                _prune_pcr0_cache()
            else:
                logger.error(f"[PCR0] ❌ Failed to build PCR0 for content {content_hash}")
        
        _last_content_hash = content_hash
        logger.info(f"[PCR0] ✅ Cache updated. Valid PCR0s: {len(_pcr0_cache)}")
        
    except Exception as e:
        logger.exception(f"[PCR0] Error in check_and_build: {e}")
    finally:
        _build_in_progress = False


async def build_pcr0_for_recent_commits(num_commits: int = None):
    """
    Build PCR0s for the last N commits on startup.
    
    This ensures that even after gateway restart, validators running
    slightly older code versions are still accepted.
    
    LOGIC:
    1. Get the last N commits that touched monitored files
    2. For each commit (newest to oldest):
       - Checkout that version
       - Compute content hash
       - If not already cached, build PCR0
       - Cache it
    3. Return to HEAD after done
    
    This is called ONCE on startup, not on every check interval.
    """
    global _pcr0_cache, _build_in_progress
    
    if num_commits is None:
        num_commits = PCR0_CACHE_SIZE
    
    if _build_in_progress:
        logger.info("[PCR0] Build already in progress, skipping historical build")
        return
    
    _build_in_progress = True
    repo_dir = BUILD_DIR
    
    try:
        logger.info(f"[PCR0] ========================================")
        logger.info(f"[PCR0] Building PCR0s for last {num_commits} commits...")
        print(f"[PCR0] Building PCR0s for last {num_commits} commits (startup cache warming)...")
        
        # Clone/update repo first
        if not await clone_or_update_repo(repo_dir):
            logger.error("[PCR0] Failed to clone/update repo for historical build")
            return
        
        # Get commits that touched monitored files
        # Use git log with path filter for monitored files
        monitored_paths = list(MONITORED_FILES) + [d.rstrip('/') for d in MONITORED_DIRS]
        path_args = ["--"] + monitored_paths
        
        proc = await asyncio.create_subprocess_exec(
            "git", "log", f"-{num_commits * 2}",  # Get more to filter
            "--format=%H|%ct|%s|%ai",
            *path_args,
            cwd=repo_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            logger.warning(f"[PCR0] git log with path filter failed, falling back to regular log")
            # Fallback: just get last N commits
            commits = await get_latest_commits(repo_dir, num_commits)
        else:
            commits = []
            for line in stdout.decode().strip().split("\n"):
                if "|" in line:
                    parts = line.split("|", 3)
                    commits.append({
                        "hash": parts[0],
                        "timestamp": parts[1] if len(parts) > 1 else "",
                        "message": parts[2] if len(parts) > 2 else "",
                        "date": parts[3] if len(parts) > 3 else "",
                    })
            commits = commits[:num_commits]  # Limit to requested count
        
        if not commits:
            logger.warning("[PCR0] No commits found for historical build")
            return
        
        logger.info(f"[PCR0] Found {len(commits)} commits to process")
        
        # Store original HEAD
        proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD",
            cwd=repo_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        original_head = stdout.decode().strip()
        
        built_count = 0
        
        for i, commit in enumerate(commits):
            commit_hash = commit["hash"]
            commit_msg = commit["message"][:50]
            
            logger.info(f"[PCR0] [{i+1}/{len(commits)}] Processing commit {commit_hash[:8]}: {commit_msg}")
            print(f"[PCR0] [{i+1}/{len(commits)}] Processing commit {commit_hash[:8]}: {commit_msg}")
            
            # Checkout this commit
            proc = await asyncio.create_subprocess_exec(
                "git", "checkout", commit_hash,
                cwd=repo_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                logger.warning(f"[PCR0] Failed to checkout {commit_hash[:8]}: {stderr.decode()}")
                continue
            
            # Compute content hash for this version
            content_hash = compute_files_content_hash(repo_dir)
            if not content_hash:
                logger.warning(f"[PCR0] Failed to compute content hash for {commit_hash[:8]}")
                continue

            if not await ensure_base_image_exists(repo_dir):
                logger.warning(f"[PCR0] Failed to prepare base image for {commit_hash[:8]}")
                continue

            cache_key = build_pcr0_cache_key(content_hash, repo_dir)
            
            # Check if already cached
            if cache_key in _pcr0_cache:
                logger.info(f"[PCR0] Cache key {cache_key} already cached, skipping")
                continue
            
            # Build PCR0 for this version
            logger.info(f"[PCR0] Building PCR0 for {commit_hash[:8]} (content: {content_hash})...")
            print(f"[PCR0] Building PCR0 for {commit_hash[:8]}...")
            
            pcr0 = await build_enclave_and_extract_pcr0(repo_dir)
            
            if pcr0:
                async with _cache_lock:
                    _pcr0_cache[cache_key] = {
                        "pcr0": pcr0,
                        "content_hash": content_hash,
                        "cache_key": cache_key,
                        "base_image_stamp": format_base_image_stamp(repo_dir),
                        "commit_hash": commit_hash,
                        "commit_timestamp": commit.get("timestamp", ""),
                        "built_at": datetime.utcnow().isoformat(),
                    }
                built_count += 1
                logger.info(f"[PCR0] ✅ Cached PCR0 for {commit_hash[:8]}: {pcr0[:32]}...")
                print(f"[PCR0] ✅ Cached PCR0 for {commit_hash[:8]}: {pcr0[:32]}...")
            else:
                logger.warning(f"[PCR0] ❌ Failed to build PCR0 for {commit_hash[:8]}")
            
            # Stop if we have enough
            if len(_pcr0_cache) >= PCR0_CACHE_SIZE:
                logger.info(f"[PCR0] Cache full ({PCR0_CACHE_SIZE} entries), stopping")
                break
        
        # Return to original HEAD
        proc = await asyncio.create_subprocess_exec(
            "git", "checkout", original_head,
            cwd=repo_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        
        logger.info(f"[PCR0] ========================================")
        logger.info(f"[PCR0] Historical build complete: {built_count} new PCR0s cached")
        logger.info(f"[PCR0] Total cached: {len(_pcr0_cache)} versions")
        print(f"[PCR0] ✅ Startup cache warming complete: {len(_pcr0_cache)} PCR0s cached")
        
    except Exception as e:
        logger.exception(f"[PCR0] Error in historical build: {e}")
    finally:
        _build_in_progress = False
        # Ensure we're back on HEAD
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "checkout", GITHUB_BRANCH,
                cwd=repo_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
        except:
            pass


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
    
    if PCR0_STARTUP_WARM_DELAY_SECONDS > 0:
        logger.info(
            "[PCR0] Delaying startup cache warming for %ss so gateway health/readiness can come up first",
            PCR0_STARTUP_WARM_DELAY_SECONDS,
        )
        await asyncio.sleep(PCR0_STARTUP_WARM_DELAY_SECONDS)

    # STARTUP: Build PCR0s for last N commits (cache warming). This is delayed
    # and optional so a gateway restart does not immediately depend on Docker
    # overlay health; static allowlist verification is still available.
    if PCR0_STARTUP_HISTORICAL_WARM_ENABLED:
        logger.info("[PCR0] Running startup cache warming (last N commits)...")
        await build_pcr0_for_recent_commits(PCR0_CACHE_SIZE)
    else:
        logger.info("[PCR0] Startup historical cache warming disabled by env")
    
    logger.info(f"[PCR0] Startup complete. Cache status: {get_cache_status()}")
    
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
    print(f"   Startup warm delay: {PCR0_STARTUP_WARM_DELAY_SECONDS}s")
    print(f"   Mode: PINNED DOCKERFILE (reproducible builds)")
    print(f"   TTL: None (pinned builds are deterministic)")


# =============================================================================
# API for verification
# =============================================================================

def verify_pcr0(pcr0: str) -> Dict:
    """
    Verify a PCR0 value against our computed cache.
    
    The cache stores PCR0 values keyed by monitored source content and base-image identity.
    This means:
    - Same code + same base image = same PCR0 (regardless of how many commits)
    - Recent build identities are cached for rolling validator compatibility
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
    for cache_key, entry in _pcr0_cache.items():
        if entry["pcr0"] == pcr0:
            return {
                "valid": True,
                "commit_hash": entry.get("commit_hash", "unknown"),
                "content_hash": entry.get("content_hash", cache_key),
                "base_image_stamp": entry.get("base_image_stamp"),
                "built_at": entry.get("built_at"),
                "message": f"PCR0 matches cache key {cache_key} (commit {entry.get('commit_hash', 'unknown')[:8]})",
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
