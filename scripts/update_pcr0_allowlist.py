#!/usr/bin/env python3
"""
Update PCR0 Allowlist

This script updates the pcr0_allowlist.json file with new PCR0 values.
Designed for use in CI/CD pipelines after building new enclave images.

Usage:
    # Add a new gateway PCR0:
    python scripts/update_pcr0_allowlist.py gateway <pcr0_hex> --version v1.0.1 --notes "Bug fix release"
    
    # Add a new validator PCR0:
    python scripts/update_pcr0_allowlist.py validator <pcr0_hex> --version v1.0.1
    
    # Remove old PCR0 values (keep only last N):
    python scripts/update_pcr0_allowlist.py gateway <pcr0_hex> --keep-last 3

Security Notes:
    - This script should only be run by the CI/CD pipeline
    - The pcr0_allowlist.json should be committed and pushed to GitHub
    - Gateway and auditors fetch the allowlist from GitHub to verify PCR0
    - Validators CANNOT fake PCR0 - it's measured by AWS Nitro hardware
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


def load_allowlist(path: Path) -> dict:
    """Load the PCR0 allowlist from file."""
    if not path.exists():
        return {
            "_comment": "PCR0 allowlist for LeadPoet TEE verification",
            "_updated": None,
            "_repo": "https://github.com/LeadPoet/Bittensor-subnet",
            "gateway_pcr0": [],
            "validator_pcr0": [],
        }
    
    with open(path) as f:
        return json.load(f)


def save_allowlist(path: Path, data: dict) -> None:
    """Save the PCR0 allowlist to file."""
    data["_updated"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')  # Trailing newline


def add_pcr0(
    allowlist: dict,
    role: str,
    pcr0: str,
    version: str,
    notes: str = None,
    keep_last: int = None,
) -> bool:
    """
    Add a new PCR0 to the allowlist.
    
    Returns:
        True if PCR0 was added, False if it already exists.
    """
    # Validate PCR0 format (SHA-384 = 96 hex chars)
    if len(pcr0) != 96:
        raise ValueError(f"PCR0 must be 96 hex characters, got {len(pcr0)}")
    try:
        bytes.fromhex(pcr0)
    except ValueError:
        raise ValueError("PCR0 must be valid hexadecimal")
    
    key = f"{role}_pcr0"
    if key not in allowlist:
        allowlist[key] = []
    
    # Check if PCR0 already exists
    existing_pcr0s = {entry["pcr0"] for entry in allowlist[key]}
    if pcr0 in existing_pcr0s:
        print(f"⚠️ PCR0 already exists in {key}")
        return False
    
    # Add new entry
    entry = {
        "pcr0": pcr0,
        "version": version,
        "deployed": datetime.utcnow().strftime("%Y-%m-%d"),
    }
    if notes:
        entry["notes"] = notes
    
    allowlist[key].append(entry)
    print(f"✅ Added PCR0 to {key}: {pcr0[:32]}...")
    
    # Prune old entries if requested
    if keep_last and len(allowlist[key]) > keep_last:
        removed = len(allowlist[key]) - keep_last
        allowlist[key] = allowlist[key][-keep_last:]
        print(f"🗑️ Removed {removed} old PCR0 value(s), keeping last {keep_last}")
    
    return True


def list_pcr0s(allowlist: dict) -> None:
    """Print all PCR0 values in the allowlist."""
    print("\n" + "=" * 70)
    print("Current PCR0 Allowlist")
    print("=" * 70)
    
    for role in ["gateway", "validator"]:
        key = f"{role}_pcr0"
        entries = allowlist.get(key, [])
        
        print(f"\n{role.upper()} ({len(entries)} entries):")
        for entry in entries:
            pcr0 = entry["pcr0"]
            version = entry.get("version", "?")
            deployed = entry.get("deployed", "?")
            notes = entry.get("notes", "")
            
            print(f"  - {pcr0[:24]}...{pcr0[-8:]}")
            print(f"    Version: {version}, Deployed: {deployed}")
            if notes:
                print(f"    Notes: {notes}")


def main():
    parser = argparse.ArgumentParser(description="Update PCR0 allowlist")
    parser.add_argument(
        "role",
        choices=["gateway", "validator", "list"],
        help="Role to update (gateway/validator) or 'list' to show current values"
    )
    parser.add_argument(
        "pcr0",
        nargs="?",
        help="PCR0 value (96 hex characters, SHA-384)"
    )
    parser.add_argument(
        "--version", "-v",
        default="unknown",
        help="Version string (e.g., v1.0.0)"
    )
    parser.add_argument(
        "--notes", "-n",
        help="Optional notes about this build"
    )
    parser.add_argument(
        "--keep-last", "-k",
        type=int,
        help="Keep only the last N PCR0 values"
    )
    parser.add_argument(
        "--path", "-p",
        default="pcr0_allowlist.json",
        help="Path to allowlist file"
    )
    
    args = parser.parse_args()
    
    # Resolve path relative to repo root
    if not Path(args.path).is_absolute():
        repo_root = Path(__file__).parent.parent
        allowlist_path = repo_root / args.path
    else:
        allowlist_path = Path(args.path)
    
    # Load current allowlist
    allowlist = load_allowlist(allowlist_path)
    
    if args.role == "list":
        list_pcr0s(allowlist)
        return
    
    if not args.pcr0:
        parser.error("PCR0 value is required for gateway/validator roles")
    
    # Add the new PCR0
    added = add_pcr0(
        allowlist,
        role=args.role,
        pcr0=args.pcr0,
        version=args.version,
        notes=args.notes,
        keep_last=args.keep_last,
    )
    
    if added:
        save_allowlist(allowlist_path, allowlist)
        print(f"💾 Saved to {allowlist_path}")
    
    # Show current state
    list_pcr0s(allowlist)


if __name__ == "__main__":
    main()

