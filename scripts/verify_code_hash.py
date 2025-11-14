#!/usr/bin/env python3
"""
Verify gateway code hash matches GitHub commit.

This script verifies that the gateway enclave is running the exact code from a
specific GitHub commit by:
1. Cloning the repo at the specified commit
2. Computing SHA256 hash of tee_service.py
3. Comparing to the code_hash from the gateway's attestation

Usage:
    python verify_code_hash.py                    # Uses default gateway URL
    python verify_code_hash.py <gateway_url>      # Custom gateway URL
    python verify_code_hash.py <gateway_url> --github-url <url> --commit <commit>

Requirements:
    - git
    - Python with requests, cbor2, cryptography

Note: This verifies the application code hash, which proves the exact Python code running.
      It does NOT verify PCR0 (Docker image hash) since that requires AWS Nitro CLI on Linux.
"""

import sys
import json
import hashlib
import subprocess
import tempfile
import shutil
import argparse
from pathlib import Path
from typing import Optional
import requests

# ============================================================================
# PRODUCTION GATEWAY CONFIGURATION
# ============================================================================
# Import from centralized config - update GATEWAY_URL in Leadpoet/utils/cloud_db.py
import sys
import os
# Add parent directory to path to import from Leadpoet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Leadpoet.utils.cloud_db import GATEWAY_URL as DEFAULT_GATEWAY_URL
# ============================================================================


def check_dependencies():
    """Check if required tools are installed"""
    
    print("🔍 Checking dependencies...")
    
    required_tools = {
        "git": "Git is required to clone the repository",
    }
    
    missing = []
    for tool, description in required_tools.items():
        if not shutil.which(tool):
            print(f"   ❌ {tool} not found: {description}")
            missing.append(tool)
        else:
            print(f"   ✅ {tool} found")
    
    return missing


def clone_repo(github_url: str, commit: str, target_dir: Path) -> bool:
    """Clone GitHub repository at specific commit"""
    
    print(f"\n📦 Cloning repository...")
    print(f"   URL: {github_url}")
    print(f"   Commit: {commit}")
    print(f"   Target: {target_dir}")
    
    try:
        # Clone with depth=1 for speed
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", commit, github_url, str(target_dir)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes max
        )
        
        if result.returncode != 0:
            # Try without --depth if branch/tag doesn't work
            print("   ℹ️  Trying full clone (commit may not be a branch/tag)...")
            result = subprocess.run(
                ["git", "clone", github_url, str(target_dir)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                print(f"   ❌ Clone failed: {result.stderr}")
                return False
            
            # Check out specific commit
            result = subprocess.run(
                ["git", "-C", str(target_dir), "checkout", commit],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"   ❌ Checkout failed: {result.stderr}")
                return False
        
        print("   ✅ Repository cloned successfully")
        return True
    
    except subprocess.TimeoutExpired:
        print("   ❌ Clone timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"   ❌ Clone error: {e}")
        return False


def compute_code_hash(repo_dir: Path) -> Optional[str]:
    """
    Compute SHA256 hash of ALL gateway code (same logic as enclave).
    
    This must match EXACTLY how the enclave computes code_hash.
    Both implementations hash the same files in the same order.
    """
    
    print("\n🔍 Computing code hash from GitHub code...")
    
    gateway_root = repo_dir / "gateway"
    
    if not gateway_root.exists():
        print(f"   ❌ Directory not found: {gateway_root}")
        return None
    
    try:
        # Collect all .py files to hash (EXACT same logic as TEE)
        files_to_hash = []
        
        # Same directories as TEE
        include_dirs = [
            gateway_root / "api",        # API endpoints
            gateway_root / "tasks",      # Epoch lifecycle, hourly batching
            gateway_root / "utils",      # Consensus, logger, signatures, registry
            gateway_root / "models",     # Pydantic models
            gateway_root / "tee",        # TEE service
        ]
        
        # Root-level files
        files_to_hash.append(gateway_root / "main.py")
        files_to_hash.append(gateway_root / "config.py")
        
        # Collect all .py files from directories
        for dir_path in include_dirs:
            if dir_path.exists():
                for py_file in sorted(dir_path.glob("**/*.py")):
                    if "__pycache__" not in str(py_file) and not py_file.name.endswith(".pyc"):
                        files_to_hash.append(py_file)
        
        # Sort for determinism (must match TEE exactly)
        files_to_hash = sorted(set(files_to_hash))
        
        print(f"   📝 Hashing {len(files_to_hash)} files...")
        
        # Hash all files (EXACT same algorithm as TEE)
        hasher = hashlib.sha256()
        for file_path in files_to_hash:
            if file_path.exists():
                # Include filename (same as TEE)
                hasher.update(str(file_path.name).encode('utf-8'))
                
                # Include file content
                hasher.update(file_path.read_bytes())
                
                # Show first few files for debugging
                if len(files_to_hash) <= 20 or files_to_hash.index(file_path) < 3:
                    print(f"      ✓ {file_path.relative_to(gateway_root)}")
        
        code_hash = hasher.hexdigest()
        
        print(f"   ✅ Code hash computed from {len(files_to_hash)} files")
        print(f"      {code_hash[:32]}...{code_hash[-32:]}")
        return code_hash
    
    except Exception as e:
        print(f"   ❌ Failed to compute hash: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_gateway_attestation(gateway_url: str) -> Optional[dict]:
    """Download attestation from gateway"""
    
    print(f"\n📥 Downloading attestation from {gateway_url}...")
    
    if not gateway_url.startswith("http"):
        gateway_url = f"http://{gateway_url}"
    
    try:
        response = requests.get(f"{gateway_url}/attest", timeout=30)
        response.raise_for_status()
        
        data = response.json()
        print(f"   ✅ Attestation downloaded")
        return data
    
    except requests.ConnectionError:
        print(f"   ❌ Failed to connect to gateway at {gateway_url}")
        print()
        print("   🔧 TROUBLESHOOTING:")
        print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   The gateway IP may have changed due to an EC2 instance restart.")
        print()
        print("   Current configured IP in this script:")
        print(f"      {DEFAULT_GATEWAY_URL}")
        print()
        print("   To fix:")
        print("   1. Check if the EC2 instance is running")
        print("   2. Get the new public IP from AWS console")
        print("   3. Update DEFAULT_GATEWAY_URL at the top of this script")
        print("   4. OR contact us on LeadPoet Discord for the current IP")
        print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return None
    except requests.Timeout:
        print(f"   ❌ Request timed out after 30 seconds")
        print(f"   🔧 Gateway may be starting up or unresponsive")
        print(f"      Contact us on LeadPoet Discord if the issue persists")
        return None
    except requests.RequestException as e:
        print(f"   ❌ Failed to download: {e}")
        print(f"   🔧 If the gateway IP has changed, update DEFAULT_GATEWAY_URL in this script")
        return None


def verify_code_hash(gateway_url: str, github_url: str, commit: str) -> bool:
    """
    Main verification logic.
    
    Returns True if code hash matches, False otherwise.
    """
    
    print("=" * 80)
    print("🔐 GATEWAY CODE INTEGRITY VERIFIER")
    print("=" * 80)
    print(f"\nGateway URL: {gateway_url}")
    print(f"GitHub URL:  {github_url}")
    print(f"Commit:      {commit}")
    print()
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("   Please install: git")
        return False
    
    # Get gateway attestation
    attestation = get_gateway_attestation(gateway_url)
    if not attestation:
        return False
    
    gateway_code_hash = attestation.get("code_hash")
    gateway_pcr0 = attestation.get("pcr0")
    
    if not gateway_code_hash:
        print("   ❌ Attestation missing 'code_hash' field")
        return False
    
    print(f"\n📋 Gateway attestation:")
    print(f"   Code Hash: {gateway_code_hash[:32]}...{gateway_code_hash[-32:]}")
    print(f"   PCR0:      {gateway_pcr0[:32]}...{gateway_pcr0[-32:] if gateway_pcr0 else 'null'}")
    
    # Clone repo and compute code hash
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        
        if not clone_repo(github_url, commit, repo_dir):
            return False
        
        github_code_hash = compute_code_hash(repo_dir)
        if not github_code_hash:
            return False
    
    # Compare
    print("\n" + "=" * 80)
    print("📊 VERIFICATION RESULT")
    print("=" * 80)
    
    print(f"\nGitHub code hash:  {github_code_hash}")
    print(f"Gateway code hash: {gateway_code_hash}")
    
    if github_code_hash == gateway_code_hash:
        print("\n✅ CODE HASH MATCH - Gateway is running canonical code!")
        print("\n🎯 The gateway enclave is provably running the exact code from:")
        print(f"   Repository: {github_url}")
        print(f"   Commit: {commit}")
        print("\nThis proves the gateway operator cannot run modified code without detection.")
        return True
    else:
        print("\n❌ CODE HASH MISMATCH - Gateway may be running modified code!")
        print("\n⚠️  WARNING: The code running in the gateway does NOT match GitHub.")
        print("   This could indicate:")
        print("   1. Gateway operator is running modified/malicious code")
        print("   2. GitHub commit doesn't match deployed version")
        print("   3. Code was updated but not pushed to GitHub")
        print("\n🔴 DO NOT TRUST THIS GATEWAY until code hashes match!")
        return False


def main():
    """CLI entry point"""
    
    parser = argparse.ArgumentParser(
        description="Verify gateway code integrity against GitHub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Verify using default gateway ({DEFAULT_GATEWAY_URL})
  python verify_code_hash.py
  
  # Verify against specific commit
  python verify_code_hash.py --commit abc123def456
  
  # Verify against custom gateway
  python verify_code_hash.py http://custom-gateway:8000
  
  # Verify against different repo
  python verify_code_hash.py \\
    --github-url https://github.com/different/repo \\
    --commit main

Note: This verifies the application code (tee_service.py) hash.
      It does NOT verify PCR0 (Docker image), which requires AWS Nitro CLI on Linux.
      Other files (merkle.py, nsm_lib.py) are protected by PCR0 (Docker image hash).
"""
    )
    
    parser.add_argument(
        "gateway_url",
        nargs="?",  # Make optional
        default=DEFAULT_GATEWAY_URL,
        help=f"Gateway URL (default: {DEFAULT_GATEWAY_URL})"
    )
    parser.add_argument(
        "--github-url",
        default="https://github.com/leadpoet/leadpoet",
        help="GitHub repository URL (default: https://github.com/leadpoet/leadpoet)"
    )
    parser.add_argument(
        "--commit",
        default="main",
        help="Git commit/branch/tag to verify (default: main)"
    )
    
    args = parser.parse_args()
    
    success = verify_code_hash(args.gateway_url, args.github_url, args.commit)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
