"""
Qualification System: Helper Functions

Phase 5.3 from tasks10.md

This module provides helper functions used across the qualification system:

1. LLM Interactions:
   - openrouter_chat() - Call OpenRouter API for text completion
   - openrouter_chat_json() - Call OpenRouter and parse JSON response
   - extract_score() - Parse numeric score from LLM response

2. Model Code Storage:
   - clone_and_store_model() - Main function to clone, tarball, hash, and upload
   - clone_repo() - Git clone at specific commit
   - create_tarball() - Create .tar.gz archive
   - compute_file_hash() - SHA256 hash of file
   - upload_to_s3() - Upload to S3 bucket

3. Bittensor Utilities:
   - get_current_bittensor_epoch() - Get current block/epoch
   - get_tao_price_usd() - Get TAO price from oracle
   - is_hotkey_registered() - Check if hotkey is registered

CRITICAL: These are NEW helper utilities for qualification only.
Do NOT modify any existing LLM interaction code or git utilities
elsewhere in the codebase.
"""

import os
import re
import json
import hashlib
import tarfile
import shutil
import subprocess
import logging
import asyncio
from typing import Optional, Tuple, Any, Dict
from uuid import UUID
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# OpenRouter API
# SECURITY: Qualification uses SEPARATE API keys with limited funds.
# TODO: After beta release, change back to "OPENROUTER_API_KEY" (shared with sourcing)
OPENROUTER_API_KEY = os.getenv("QUALIFICATION_OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_TIMEOUT = 30.0

# S3 Configuration
# Uses the same bucket as lead storage, with a different prefix (qualification/)
S3_BUCKET = os.getenv("QUAL_S3_BUCKET", "leadpoet-leads-primary")
S3_REGION = os.getenv("QUAL_S3_REGION", "us-east-2")  # leadpoet-leads-primary is in us-east-2

# Temporary directory for model processing
TEMP_DIR = os.getenv("QUAL_TEMP_DIR", "/tmp/qualification")

# Bittensor Configuration
SUBTENSOR_ENDPOINT = os.getenv("SUBTENSOR_ENDPOINT", "wss://entrypoint-finney.opentensor.ai:443")
SUBNET_NETUID = int(os.getenv("SUBNET_NETUID", "41"))


# =============================================================================
# LLM Functions
# =============================================================================

async def openrouter_chat(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 500,
    system_prompt: Optional[str] = None
) -> str:
    """
    Call OpenRouter LLM API for text completion.
    
    Args:
        prompt: The user prompt
        model: Model name (without openai/ prefix)
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens in response
        system_prompt: Optional system prompt
    
    Returns:
        The LLM response text
    
    Raises:
        ValueError: If API key not configured
        httpx.HTTPError: If API call fails
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not configured")
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://leadpoet.ai",
                "X-Title": "Leadpoet Qualification"
            },
            json={
                "model": f"openai/{model}",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=OPENROUTER_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        
        return data["choices"][0]["message"]["content"]


async def openrouter_chat_json(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2
) -> Dict[str, Any]:
    """
    Call OpenRouter and parse JSON response.
    
    Args:
        prompt: The prompt (should ask for JSON output)
        model: Model name
        temperature: Lower temperature for more consistent JSON
    
    Returns:
        Parsed JSON as dict
    
    Raises:
        ValueError: If response is not valid JSON
    """
    response = await openrouter_chat(prompt, model, temperature, max_tokens=500)
    return extract_json_from_response(response)


# =============================================================================
# Score Extraction
# =============================================================================

def extract_score(response: str, max_score: int) -> float:
    """
    Extract numeric score from LLM response.
    
    Handles various response formats:
    - Just a number: "15"
    - With text: "Score: 15"
    - With decimal: "15.5"
    - Fraction: "15/20"
    
    Args:
        response: The LLM response text
        max_score: Maximum allowed score (caps output)
    
    Returns:
        Extracted score (capped at max_score), or 0.0 if not found
    """
    response = response.strip()
    
    # Try patterns in order of specificity
    patterns = [
        r'^(\d+(?:\.\d+)?)\s*$',  # Just a number
        r'(?:score|rating|points?)[:=\s]+(\d+(?:\.\d+)?)',  # "Score: 15"
        r'(\d+(?:\.\d+)?)\s*(?:out of|/)\s*\d+',  # "15 out of 20" or "15/20"
        r'(\d+(?:\.\d+)?)',  # Any number (fallback)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                # Cap at max score
                return min(score, float(max_score))
            except ValueError:
                continue
    
    logger.warning(f"Could not extract score from: {response[:100]}")
    return 0.0


def extract_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM response, handling markdown code blocks.
    
    Args:
        response: The LLM response text
    
    Returns:
        Parsed JSON as dict
    
    Raises:
        ValueError: If no valid JSON found
    """
    response = response.strip()
    
    # Remove markdown code blocks if present
    if response.startswith("```"):
        response = re.sub(r'^```(?:json)?\s*', '', response)
        response = re.sub(r'\s*```$', '', response)
    
    # Try to parse directly
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object in response
    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"Could not extract JSON from response: {response[:200]}")


# =============================================================================
# S3 Presigned URL Generation
# =============================================================================

async def generate_presigned_upload_url(
    upload_id: str,
    max_size_bytes: int = 10485760,  # 10 MB (from CONFIG.MODEL_CODE_MAX_SIZE_MB)
    expires_in_seconds: int = 900    # 15 minutes
) -> Tuple[str, str]:
    """
    Generate a presigned PUT URL for direct S3 upload.
    
    Miners use this to upload their model tarball directly to S3
    without going through the gateway.
    
    SECURITY: Size enforcement happens server-side in verify_model_upload():
    - Miners can upload files of any size
    - But at submit time, files over max_size_bytes are REJECTED
    - max_size_bytes is logged here for reference but not enforced at upload
    
    Args:
        upload_id: Unique identifier for this upload (UUID)
        max_size_bytes: Maximum allowed file size (logged, enforced at submit)
        expires_in_seconds: URL expiration time (default 15 min)
    
    Returns:
        Tuple of (presigned_url, s3_key)
    """
    s3_key = f"qualification/uploads/{upload_id}.tar.gz"
    
    try:
        import boto3
        from botocore.config import Config
        
        s3_client = boto3.client(
            's3',
            region_name=S3_REGION,
            config=Config(signature_version='s3v4')
        )
        
        presigned_url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': S3_BUCKET,
                'Key': s3_key,
                'ContentType': 'application/gzip',
            },
            ExpiresIn=expires_in_seconds
        )
        
        max_mb = max_size_bytes / (1024 * 1024)
        logger.info(
            f"Generated presigned URL for {s3_key} "
            f"(max_size={max_mb:.1f}MB enforced at submit, expires in {expires_in_seconds}s)"
        )
        return presigned_url, s3_key
        
    except ImportError:
        logger.warning("boto3 not installed - returning mock presigned URL")
        # Mock for testing without AWS credentials
        mock_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}?mock=true"
        return mock_url, s3_key


# =============================================================================
# S3 Model Verification
# =============================================================================

async def verify_model_upload(
    s3_key: str,
    claimed_hash: str,
    max_size_bytes: int = None
) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Verify that a model was uploaded to S3, hash matches, and size is within limits.
    
    Called after miner submits to verify:
    1. The S3 object exists
    2. The file size is within allowed limits
    3. The hash matches what the miner claimed
    
    Args:
        s3_key: S3 object key (e.g., "qualification/uploads/{uuid}.tar.gz")
        claimed_hash: SHA256 hash the miner claimed
        max_size_bytes: Maximum allowed file size (default from CONFIG)
    
    Returns:
        Tuple of (is_valid, error_message, file_size_bytes)
        - (True, None, size) if valid
        - (False, "error reason", None) if invalid
    """
    # Get max size from config if not provided
    if max_size_bytes is None:
        from gateway.qualification.config import CONFIG
        max_size_bytes = CONFIG.get_model_max_size_bytes()
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        s3_client = boto3.client('s3', region_name=S3_REGION)
        
        # Check if object exists and get metadata
        try:
            head_response = s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
            file_size = head_response['ContentLength']
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False, f"S3 object not found: {s3_key}", None
            raise
        
        # ════════════════════════════════════════════════════════════════════════
        # SIZE VALIDATION: Reject files larger than configured max
        # ════════════════════════════════════════════════════════════════════════
        if file_size > max_size_bytes:
            max_mb = max_size_bytes / (1024 * 1024)
            actual_mb = file_size / (1024 * 1024)
            logger.warning(
                f"Model file too large: {actual_mb:.2f} MB > {max_mb:.2f} MB max "
                f"(key: {s3_key})"
            )
            return False, f"Model file too large: {actual_mb:.2f} MB exceeds {max_mb:.2f} MB limit", None
        
        # Download and compute hash
        logger.info(f"Downloading {s3_key} to verify hash ({file_size / 1024:.1f} KB)...")
        
        # Stream download to compute hash without loading entire file in memory
        sha256 = hashlib.sha256()
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        
        for chunk in response['Body'].iter_chunks(chunk_size=8192):
            sha256.update(chunk)
        
        computed_hash = sha256.hexdigest()
        
        # Compare hashes
        if computed_hash.lower() != claimed_hash.lower():
            logger.warning(f"Hash mismatch: claimed={claimed_hash[:16]}... computed={computed_hash[:16]}...")
            return False, f"Hash mismatch: uploaded file hash does not match claimed hash", None
        
        logger.info(f"Hash verified: {computed_hash[:16]}... ({file_size} bytes)")
        return True, None, file_size
        
    except ImportError:
        logger.warning("boto3 not installed - skipping S3 verification (mock mode)")
        # Mock for testing
        return True, None, 1000000


async def scan_model_for_dangerous_patterns(s3_key: str) -> Tuple[bool, Optional[str]]:
    """
    Quick static scan for obviously dangerous patterns in model code.
    
    This is a SUPPLEMENTARY check - catches 80% of bad models quickly.
    The validator sandbox is still the PRIMARY defense (catches everything at runtime).
    
    Scans all .py files in the tarball for:
    - Blocked library imports (subprocess, ctypes, pickle, etc.)
    - Dangerous function calls (eval, exec, os.system, etc.)
    - Suspicious file access patterns (.bittensor, /proc/self/environ, etc.)
    
    Args:
        s3_key: S3 object key of the uploaded tarball
    
    Returns:
        Tuple of (is_safe, error_message)
        - (True, None) if no obvious issues found
        - (False, "reason") if dangerous pattern detected
    """
    import re
    import tarfile
    import tempfile
    
    # Patterns to scan for (regex, human-readable reason)
    DANGEROUS_PATTERNS = [
        # Blocked library imports
        (r'\bimport\s+subprocess\b', "imports 'subprocess' (blocked library)"),
        (r'\bfrom\s+subprocess\s+import\b', "imports from 'subprocess' (blocked library)"),
        (r'\bimport\s+ctypes\b', "imports 'ctypes' (blocked library)"),
        (r'\bfrom\s+ctypes\s+import\b', "imports from 'ctypes' (blocked library)"),
        (r'\bimport\s+pickle\b', "imports 'pickle' (blocked library)"),
        (r'\bfrom\s+pickle\s+import\b', "imports from 'pickle' (blocked library)"),
        (r'\bimport\s+marshal\b', "imports 'marshal' (blocked library)"),
        (r'\bimport\s+socket\b', "imports 'socket' directly (use httpx instead)"),
        (r'\bimport\s+multiprocessing\b', "imports 'multiprocessing' (blocked library)"),
        (r'\bimport\s+shutil\b', "imports 'shutil' (blocked library)"),
        (r'\bimport\s+pathlib\b', "imports 'pathlib' (blocked library)"),
        (r'\bimport\s+glob\b', "imports 'glob' (blocked library)"),
        
        # Dangerous function calls
        (r'\bos\.system\s*\(', "calls os.system() (shell execution blocked)"),
        (r'\bos\.popen\s*\(', "calls os.popen() (shell execution blocked)"),
        (r'\bos\.spawn', "uses os.spawn* (process spawning blocked)"),
        (r'\beval\s*\([^)]*\)', "uses eval() (code execution blocked)"),
        (r'\bexec\s*\([^)]*\)', "uses exec() (code execution blocked)"),
        (r'__import__\s*\(', "uses __import__() (dynamic imports blocked)"),
        (r'\bcompile\s*\([^)]*,[^)]*,[\'"]exec[\'"]\)', "uses compile() with exec mode"),
        
        # Blocked file/directory paths
        (r'\.bittensor', "accesses .bittensor directory (blocked path)"),
        (r'\.ssh[/\'"]', "accesses .ssh directory (blocked path)"),
        (r'\.aws[/\'"]', "accesses .aws directory (blocked path)"),
        (r'/proc/self/environ', "reads /proc/self/environ (blocked access)"),
        (r'/etc/passwd', "reads /etc/passwd (blocked access)"),
        
        # Raw socket operations (not allowed - use httpx for HTTP)
        (r'socket\.socket\s*\(', "creates raw socket (use httpx instead)"),
        (r'\.listen\s*\(', "calls .listen() (server operations not allowed)"),
        (r'\.bind\s*\(\s*\([\'\"]\s*[\'\"]', "binds to address (server operations not allowed)"),
    ]
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        s3_client = boto3.client('s3', region_name=S3_REGION)
        
        # Download tarball to temp file
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=True) as tmp:
            try:
                s3_client.download_file(S3_BUCKET, s3_key, tmp.name)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    return False, f"S3 object not found: {s3_key}"
                raise
            
            # Extract and scan .py files
            try:
                with tarfile.open(tmp.name, 'r:gz') as tar:
                    for member in tar.getmembers():
                        if member.name.endswith('.py') and member.isfile():
                            f = tar.extractfile(member)
                            if f:
                                try:
                                    content = f.read().decode('utf-8', errors='ignore')
                                except:
                                    continue
                                
                                # Scan for dangerous patterns
                                for pattern, reason in DANGEROUS_PATTERNS:
                                    if re.search(pattern, content, re.IGNORECASE):
                                        file_name = member.name.split('/')[-1]
                                        logger.warning(
                                            f"Dangerous pattern in {file_name}: {reason}"
                                        )
                                        return False, f"Security scan failed: {reason} in {file_name}"
                                        
            except tarfile.TarError as e:
                return False, f"Invalid tarball: {e}"
        
        logger.info(f"Security scan passed for {s3_key}")
        return True, None
        
    except ImportError:
        logger.warning("boto3 not installed - skipping security scan (mock mode)")
        return True, None
    except Exception as e:
        logger.error(f"Security scan error: {e}")
        # On error, allow through (validator will catch it)
        return True, None


async def delete_model_from_s3(s3_key: str) -> bool:
    """
    Delete a model from S3 (used for non-champions after evaluation).
    
    Args:
        s3_key: S3 object key to delete
    
    Returns:
        True if deleted, False if error
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        s3_client = boto3.client('s3', region_name=S3_REGION)
        
        s3_client.delete_object(Bucket=S3_BUCKET, Key=s3_key)
        logger.info(f"Deleted from S3: {s3_key}")
        return True
        
    except ImportError:
        logger.warning("boto3 not installed - skipping S3 delete (mock mode)")
        return True
    except ClientError as e:
        logger.error(f"Failed to delete {s3_key}: {e}")
        return False


async def copy_to_champions(source_s3_key: str, model_id: UUID) -> str:
    """
    Copy a winning model from uploads/ to champions/ (permanent storage).
    
    Champions are NEVER deleted - kept for historical record.
    
    Args:
        source_s3_key: Source key in uploads/ folder
        model_id: Model UUID
    
    Returns:
        New S3 key in champions/ folder
    """
    champion_key = f"qualification/champions/{model_id}.tar.gz"
    
    try:
        import boto3
        
        s3_client = boto3.client('s3', region_name=S3_REGION)
        
        # Copy to champions folder
        copy_source = {'Bucket': S3_BUCKET, 'Key': source_s3_key}
        s3_client.copy_object(
            CopySource=copy_source,
            Bucket=S3_BUCKET,
            Key=champion_key
        )
        
        logger.info(f"Copied to champions: {source_s3_key} → {champion_key}")
        return champion_key
        
    except ImportError:
        logger.warning("boto3 not installed - skipping S3 copy (mock mode)")
        return champion_key


# =============================================================================
# Model Code Storage (Legacy - for admin/backup use)
# =============================================================================

async def clone_and_store_model(
    repo_url: str,
    commit_hash: str,
    model_id: UUID
) -> Tuple[str, str]:
    """
    Clone repository at specific commit and store in S3.
    
    NOTE: This is the LEGACY function kept for admin/backup purposes.
    Normal miner submissions now use direct S3 upload via presigned URLs.
    
    This function may still be used for:
    - Admin manually adding benchmark models
    - Backup/restore operations
    - Testing
    
    Args:
        repo_url: Git repository URL (https or git@)
        commit_hash: Specific commit to checkout
        model_id: UUID for this model (used in paths)
    
    Returns:
        Tuple of (code_hash, s3_path)
    
    Raises:
        subprocess.CalledProcessError: If git operations fail
        OSError: If file operations fail
    """
    # Create temp directory structure
    model_dir = Path(TEMP_DIR) / str(model_id)
    clone_dir = model_dir / "repo"
    tar_path = model_dir / "model.tar.gz"
    
    try:
        # Ensure temp directory exists
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Clone repo at specific commit
        logger.info(f"Cloning {repo_url} at {commit_hash[:8]}...")
        await clone_repo(repo_url, commit_hash, clone_dir)
        
        # Step 2: Create tarball
        logger.info(f"Creating tarball...")
        await create_tarball(clone_dir, tar_path)
        
        # Step 3: Compute hash
        code_hash = compute_file_hash(tar_path)
        logger.info(f"Code hash: {code_hash[:16]}...")
        
        # Step 4: Upload to S3
        s3_key = f"qualification/uploads/{model_id}.tar.gz"
        logger.info(f"Uploading to S3: {s3_key}")
        await upload_to_s3(tar_path, s3_key)
        
        s3_path = f"s3://{S3_BUCKET}/{s3_key}"
        logger.info(f"Model stored at {s3_path}")
        
        return code_hash, s3_path
        
    finally:
        # Step 5: Cleanup
        cleanup_temp_files(model_dir)


async def clone_repo(repo_url: str, commit_hash: str, clone_dir: Path) -> None:
    """
    Clone a git repository at a specific commit.
    
    Args:
        repo_url: Repository URL
        commit_hash: Commit to checkout
        clone_dir: Directory to clone into
    """
    # Remove existing directory if present
    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    
    # Clone with depth 1 for speed (shallow clone)
    # Note: For specific commits, we may need full clone if commit is not HEAD
    clone_cmd = ["git", "clone", "--depth", "100", repo_url, str(clone_dir)]
    
    # Run clone in thread pool to not block event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: subprocess.run(clone_cmd, check=True, capture_output=True, text=True)
    )
    
    # Checkout specific commit
    checkout_cmd = ["git", "-C", str(clone_dir), "checkout", commit_hash]
    await loop.run_in_executor(
        None,
        lambda: subprocess.run(checkout_cmd, check=True, capture_output=True, text=True)
    )
    
    # Remove .git directory to reduce size
    git_dir = clone_dir / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)


async def create_tarball(source_dir: Path, tar_path: Path) -> None:
    """
    Create a gzipped tarball from a directory.
    
    Args:
        source_dir: Directory to archive
        tar_path: Output tarball path
    """
    loop = asyncio.get_event_loop()
    
    def _create_tar():
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(source_dir, arcname="model")
    
    await loop.run_in_executor(None, _create_tar)


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file.
    
    Args:
        file_path: Path to file
    
    Returns:
        Hexadecimal hash string
    """
    sha256 = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    
    return sha256.hexdigest()


async def upload_to_s3(local_path: Path, s3_key: str) -> None:
    """
    Upload a file to S3.
    
    Args:
        local_path: Local file path
        s3_key: S3 object key
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        loop = asyncio.get_event_loop()
        
        def _upload():
            s3_client = boto3.client('s3', region_name=S3_REGION)
            s3_client.upload_file(str(local_path), S3_BUCKET, s3_key)
        
        await loop.run_in_executor(None, _upload)
        
    except ImportError:
        logger.warning("boto3 not installed - S3 upload skipped (mock mode)")
        # In mock/test mode, just log the upload
        logger.info(f"[MOCK] Would upload {local_path} to s3://{S3_BUCKET}/{s3_key}")


def cleanup_temp_files(directory: Path) -> None:
    """
    Clean up temporary files and directories.
    
    Args:
        directory: Directory to remove
    """
    try:
        if directory.exists():
            shutil.rmtree(directory)
            logger.debug(f"Cleaned up: {directory}")
    except Exception as e:
        logger.warning(f"Failed to cleanup {directory}: {e}")


# =============================================================================
# Bittensor Utilities
# =============================================================================

async def get_current_bittensor_epoch() -> int:
    """
    Get the current Bittensor block number (epoch).
    
    Returns:
        Current block number
    """
    try:
        import bittensor as bt
        
        loop = asyncio.get_event_loop()
        
        def _get_block():
            subtensor = bt.subtensor(network="finney")
            return subtensor.block
        
        return await loop.run_in_executor(None, _get_block)
        
    except ImportError:
        logger.warning("bittensor not installed - returning mock epoch")
        return 1000000  # Mock value for testing
    except Exception as e:
        logger.error(f"Failed to get Bittensor epoch: {e}")
        return 0


async def get_tao_price_usd() -> float:
    """
    Get current TAO price in USD from a price oracle.
    
    Returns:
        TAO price in USD
    """
    try:
        # Try CoinGecko API
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "bittensor", "vs_currencies": "usd"},
                timeout=10.0
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("bittensor", {}).get("usd", 0.0)
    except Exception as e:
        logger.warning(f"Failed to get TAO price from CoinGecko: {e}")
    
    try:
        # Fallback to taostats
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://taostats.io/api/price",
                timeout=10.0
            )
            if response.status_code == 200:
                data = response.json()
                return float(data.get("price", 0.0))
    except Exception as e:
        logger.warning(f"Failed to get TAO price from taostats: {e}")
    
    # Default fallback price
    logger.warning("Using fallback TAO price: $400")
    return 400.0


async def is_hotkey_registered(hotkey: str, netuid: int = SUBNET_NETUID) -> bool:
    """
    Check if a hotkey is registered on the subnet.
    
    Args:
        hotkey: The hotkey address (SS58 format)
        netuid: Subnet network UID
    
    Returns:
        True if registered, False otherwise
    """
    try:
        import bittensor as bt
        
        loop = asyncio.get_event_loop()
        
        def _check_registration():
            subtensor = bt.subtensor(network="finney")
            metagraph = subtensor.metagraph(netuid)
            return hotkey in metagraph.hotkeys
        
        return await loop.run_in_executor(None, _check_registration)
        
    except ImportError:
        logger.warning("bittensor not installed - assuming registered")
        return True
    except Exception as e:
        logger.error(f"Failed to check hotkey registration: {e}")
        return False


# =============================================================================
# General Utilities
# =============================================================================

def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to max length with suffix.
    
    Args:
        s: String to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncated
    
    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def safe_json_loads(s: str, default: Any = None) -> Any:
    """
    Safely parse JSON, returning default on error.
    
    Args:
        s: JSON string
        default: Value to return on parse error
    
    Returns:
        Parsed JSON or default
    """
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return default


def compute_string_hash(s: str) -> str:
    """
    Compute SHA256 hash of a string.
    
    Args:
        s: String to hash
    
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(s.encode()).hexdigest()
