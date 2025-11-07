"""
Initialize MinIO bucket on gateway startup.

This script ensures the required bucket exists in MinIO before the gateway
starts accepting requests. Called during gateway startup.
"""
import sys
import time
from minio import Minio
from minio.error import S3Error

# Configuration from environment
from gateway.config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_BUCKET
)

def init_minio_bucket(max_retries=10, retry_delay=2):
    """
    Create MinIO bucket if it doesn't exist.
    
    Retries connection if MinIO is still starting up.
    """
    # Parse endpoint (remove http:// prefix if present)
    endpoint = MINIO_ENDPOINT.replace("http://", "").replace("https://", "")
    secure = MINIO_ENDPOINT.startswith("https://")
    
    print(f"🔍 Initializing MinIO bucket: {MINIO_BUCKET}")
    print(f"   Endpoint: {endpoint}")
    
    for attempt in range(1, max_retries + 1):
        try:
            # Initialize MinIO client
            client = Minio(
                endpoint,
                access_key=MINIO_ACCESS_KEY,
                secret_key=MINIO_SECRET_KEY,
                secure=secure
            )
            
            # Check if bucket exists
            if client.bucket_exists(MINIO_BUCKET):
                print(f"✅ Bucket '{MINIO_BUCKET}' already exists")
                return True
            
            # Create bucket
            client.make_bucket(MINIO_BUCKET)
            print(f"✅ Created bucket '{MINIO_BUCKET}' successfully")
            return True
            
        except S3Error as e:
            print(f"❌ S3 Error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print(f"   Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"❌ Failed to initialize MinIO after {max_retries} attempts")
                return False
                
        except Exception as e:
            print(f"❌ Unexpected error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print(f"   Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"❌ Failed to initialize MinIO after {max_retries} attempts")
                return False
    
    return False


if __name__ == "__main__":
    success = init_minio_bucket()
    sys.exit(0 if success else 1)

