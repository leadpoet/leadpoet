"""
Storage Utility
===============

Presigned URL generation and storage verification for AWS S3.

Provides integrity verification by checking SHA256 hashes match CIDs.
"""

import boto3
from botocore.client import Config
import hashlib
import sys
import os

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from gateway.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_S3_BUCKET,
    AWS_S3_REGION,
    PRESIGNED_URL_EXPIRY_SECONDS
)

# ============================================================
# Initialize S3 Client (AWS)
# ============================================================
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_S3_REGION,
    config=Config(signature_version='s3v4')
)


def generate_presigned_put_urls(cid: str) -> dict:
    """
    Generate presigned PUT URL for S3.
    
    Args:
        cid: Content identifier (SHA256 hash of lead blob)
    
    Returns:
        {
            "s3_url": "https://s3.amazonaws.com/...",
            "expires_in": 60
        }
    
    Example:
        >>> cid = "abc123def456..."
        >>> urls = generate_presigned_put_urls(cid)
        >>> # Miner uploads to S3
        >>> requests.put(urls['s3_url'], data=lead_blob)
    
    Notes:
        - Object key format: leads/{cid}.json
        - URLs expire after PRESIGNED_URL_EXPIRY_SECONDS (default 60s)
    """
    # Object key format: leads/{cid}.json
    object_key = f"leads/{cid}.json"
    
    # Generate S3 presigned URL
    s3_url = s3_client.generate_presigned_url(
        'put_object',
        Params={
            'Bucket': AWS_S3_BUCKET,
            'Key': object_key,
            'ContentType': 'application/json'
        },
        ExpiresIn=PRESIGNED_URL_EXPIRY_SECONDS
    )
    
    print(f"🔍 Presigned URL generated for S3")
    
    return {
        "s3_url": s3_url,
        "expires_in": PRESIGNED_URL_EXPIRY_SECONDS
    }


def verify_storage_proof(cid: str, mirror: str = "s3") -> bool:
    """
    Verify that blob exists in S3 storage and matches CID.
    
    Downloads the blob from S3, computes SHA256 hash,
    and verifies it matches the expected CID.
    
    Args:
        cid: Expected content hash (SHA256)
        mirror: Storage backend (only "s3" is supported)
    
    Returns:
        True if blob exists and hash matches CID
        False if blob missing, inaccessible, or hash mismatch
    
    Example:
        >>> cid = "abc123def456..."
        >>> verify_storage_proof(cid, "s3")
        True
    
    Notes:
        - This is called AFTER miner uploads to presigned URL
        - Gateway fetches and recomputes hash independently
        - Prevents blob substitution attacks
        - Used for STORAGE_PROOF event logging
    """
    object_key = f"leads/{cid}.json"
    
    try:
        if mirror != "s3":
            print(f"❌ Only S3 storage is supported (requested: {mirror})")
            return False
        
        # Fetch from S3
        response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=object_key)
        blob = response['Body'].read()
        
        # Compute SHA256 hash of downloaded blob
        computed_hash = hashlib.sha256(blob).hexdigest()
        
        # Verify hash matches expected CID
        if computed_hash != cid:
            print(f"⚠️  Hash mismatch in S3: expected {cid[:16]}..., got {computed_hash[:16]}...")
            return False
        
        print(f"✅ Storage proof verified for S3: {cid[:16]}...")
        return True
    
    except Exception as e:
        print(f"❌ Storage verification error (S3): {e}")
        return False


def check_blob_exists(cid: str, mirror: str = "s3") -> bool:
    """
    Check if blob exists in S3 storage (without downloading).
    
    Faster than verify_storage_proof() but doesn't verify hash.
    
    Args:
        cid: Content identifier
        mirror: Storage backend (only "s3" is supported)
    
    Returns:
        True if blob exists, False otherwise
    
    Example:
        >>> check_blob_exists("abc123...", "s3")
        True
    """
    object_key = f"leads/{cid}.json"
    
    try:
        if mirror != "s3":
            return False
        
        # Check if object exists in S3
        s3_client.head_object(Bucket=AWS_S3_BUCKET, Key=object_key)
        return True
    
    except Exception as e:
        # Object doesn't exist or error occurred
        return False


def delete_blob(cid: str, mirror: str = "s3") -> bool:
    """
    Delete blob from S3 storage.
    
    WARNING: Use with caution! Transparency log is append-only,
    but blobs can be deleted for GDPR compliance.
    
    Args:
        cid: Content identifier
        mirror: Storage backend (only "s3" is supported)
    
    Returns:
        True if deleted successfully
    
    Example:
        >>> delete_blob("abc123...", "s3")
        True
    """
    object_key = f"leads/{cid}.json"
    
    try:
        if mirror != "s3":
            return False
        
        s3_client.delete_object(Bucket=AWS_S3_BUCKET, Key=object_key)
        print(f"🗑️  Deleted {object_key} from S3")
        return True
    
    except Exception as e:
        print(f"❌ Delete error (S3): {e}")
        return False


def get_storage_stats() -> dict:
    """
    Get storage statistics for S3.
    
    Returns:
        {
            "s3": {
                "total_objects": 123,
                "total_size_bytes": 456789
            }
        }
    
    Example:
        >>> stats = get_storage_stats()
        >>> print(f"S3 has {stats['s3']['total_objects']} objects")
    """
    stats = {}
    
    # S3 statistics
    try:
        s3_objects = s3_client.list_objects_v2(Bucket=AWS_S3_BUCKET, Prefix="leads/")
        
        if 'Contents' in s3_objects:
            s3_count = len(s3_objects['Contents'])
            s3_size = sum(obj['Size'] for obj in s3_objects['Contents'])
        else:
            s3_count = 0
            s3_size = 0
        
        stats['s3'] = {
            "total_objects": s3_count,
            "total_size_bytes": s3_size
        }
    except Exception as e:
        print(f"❌ S3 stats error: {e}")
        stats['s3'] = {"total_objects": 0, "total_size_bytes": 0}
    
    return stats


def print_storage_stats():
    """
    Print storage statistics in a readable format.
    
    Example:
        >>> print_storage_stats()
        ============================================================
        Storage Statistics
        ============================================================
        AWS S3 (leadpoet-leads-primary):
          Objects: 123
          Size: 456.7 KB
        ============================================================
    """
    stats = get_storage_stats()
    
    print("=" * 60)
    print("Storage Statistics")
    print("=" * 60)
    
    # S3
    s3_objects = stats['s3']['total_objects']
    s3_size = stats['s3']['total_size_bytes']
    s3_size_kb = s3_size / 1024
    
    print(f"AWS S3 ({AWS_S3_BUCKET}):")
    print(f"  Objects: {s3_objects}")
    print(f"  Size: {s3_size_kb:.1f} KB")
    
    print("=" * 60)

