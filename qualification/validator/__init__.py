"""
Qualification System Validator Module

This module contains the QualificationValidator class that runs as a separate
process to execute model evaluations in TEE sandboxes.

CRITICAL: This is a completely NEW validator for qualification only.
Do NOT modify neurons/validator.py or any existing validator code.
This runs as a separate, independent process.

Components:
- main.py: QualificationValidator main loop class
- sandbox.py: TEE sandbox manager (Phase 3.2)
- db_hash.py: Database integrity hash computation (Phase 3.3)
"""

from qualification.validator.main import QualificationValidator
from qualification.validator.sandbox import (
    TEESandbox,
    SandboxError,
    SandboxStartError,
    SandboxBuildError,
    SandboxExecutionError,
    create_sandbox,
)
from qualification.validator.db_hash import (
    compute_db_hash,
    compute_db_hash_with_metadata,
    verify_db_hash,
    compute_hash_from_ids,
    DatabaseHashError,
)

__all__ = [
    # Main validator
    "QualificationValidator",
    
    # Sandbox
    "TEESandbox",
    "SandboxError",
    "SandboxStartError",
    "SandboxBuildError",
    "SandboxExecutionError",
    "create_sandbox",
    
    # Database hash
    "compute_db_hash",
    "compute_db_hash_with_metadata",
    "verify_db_hash",
    "compute_hash_from_ids",
    "DatabaseHashError",
]
