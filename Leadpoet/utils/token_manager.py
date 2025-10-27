import os
import time
import requests
import jwt as pyjwt
from datetime import datetime, timedelta
from typing import Optional
import logging
import bittensor as bt

logger = logging.getLogger(__name__)


class TokenManager:
    """
    Manages JWT token lifecycle for Supabase authentication.
    Automatically refreshes tokens before expiry to prevent service interruption.
    """
    
    def __init__(self, hotkey: str, wallet=None, token_endpoint: str = None, netuid: int = None, network: str = None):
        """
        Initialize TokenManager
        
        Args:
            hotkey: SS58 address of the hotkey
            wallet: Bittensor wallet instance (required for signing)
            token_endpoint: URL of JWT issuance endpoint
            netuid: Subnet ID (e.g., 71 for mainnet, 401 for testnet)
            network: Network name ('finney' for mainnet, 'test' for testnet)
        """
        self.hotkey = hotkey
        self.wallet = wallet
        self.token_endpoint = token_endpoint or \
            "https://qplwoislplkcegvdmbim.supabase.co/functions/v1/issue-jwt"
        self.netuid = netuid
        self.network = network
        
        self.env_var_name = f"SUPABASE_JWT_{hotkey[:8]}"
        self.token = os.getenv(self.env_var_name) or os.getenv("SUPABASE_JWT")
        
        self.token_expires = None
        self.role = None
        self.token_info = None
        self.token_hotkey = None
        
        if self.token:
            self._parse_token()
            
            if self.token_hotkey and self.token_hotkey != self.hotkey:
                bt.logging.warning(
                    f"⚠️ JWT hotkey mismatch! "
                    f"JWT hotkey: {self.token_hotkey[:10]}..., "
                    f"Wallet hotkey: {self.hotkey[:10]}..."
                )
                bt.logging.info("🔄 Requesting new JWT for current wallet...")
                self.token = None  # Clear mismatched token
                self.token_expires = None  # Force refresh
            else:
                os.environ['SUPABASE_JWT'] = self.token
                bt.logging.debug(f"✅ Loaded wallet-specific JWT for {hotkey[:10]}...")
    
    def _parse_token(self) -> bool:
        """
        Decode JWT to extract expiry timestamp, role, and hotkey without verification.
        
        Returns:
            True if parsing succeeded, False otherwise
        """
        try:
            # Decode without verification to extract claims
            decoded = pyjwt.decode(
                self.token, 
                options={"verify_signature": False, "verify_exp": False}
            )
            
            self.token_expires = datetime.fromtimestamp(decoded['exp'])
            self.role = decoded.get('app_role', 'unknown')
            self.token_hotkey = decoded.get('hotkey')  # Extract hotkey from JWT
            
            logger.info(
                f"Token parsed - Role: {self.role}, "
                f"Hotkey: {self.token_hotkey[:10] if self.token_hotkey else 'N/A'}..., "
                f"Expires: {self.token_expires}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error parsing token: {e}")
            self.token_expires = datetime.now()  # Force immediate refresh
            return False
    
    def is_expired(self) -> bool:
        """Check if token is already expired."""
        if not self.token_expires:
            return True
        return datetime.now() >= self.token_expires
    
    def needs_refresh(self, threshold_hours: int = 1) -> bool:
        """
        Check if token needs refresh based on expiry threshold.
        
        Args:
            threshold_hours: Refresh if token expires within this many hours
            
        Returns:
            True if token needs refresh
        """
        if not self.token_expires:
            return True
        
        threshold = datetime.now() + timedelta(hours=threshold_hours)
        return self.token_expires <= threshold
    
    def refresh_token(self) -> bool:
        """
        Request a new JWT token from the issuance endpoint using signature authentication.
        
        Returns:
            True if refresh succeeded, False otherwise
        """
        try:
            bt.logging.info(f"Requesting new token for hotkey: {self.hotkey[:8]}...")
            
            # Use the new signature-based authentication
            new_token = self.request_new_token()
            
            if new_token:
                # Update .env file
                if self._update_env_file(new_token):
                    # Update in-memory token
                    self.token = new_token
                    self._parse_token()
                    
                    bt.logging.info("✅ Token refreshed successfully")
                    bt.logging.info(f"   Role: {self.role}")
                    bt.logging.info(f"   Expires: {self.token_expires}")
                    if self.token_expires:
                        bt.logging.info(f"   Valid for: {(self.token_expires - datetime.now()).total_seconds() / 3600:.1f} hours")
                    
                    # Update environment variable for current process (wallet-specific)
                    os.environ[self.env_var_name] = new_token
                    os.environ['SUPABASE_JWT'] = new_token  # Also update generic for backward compat
                    
                    return True
                else:
                    bt.logging.error("Failed to update .env file")
                    return False
            else:
                bt.logging.error("Failed to get new token from request_new_token()")
                return False
                
        except Exception as e:
            bt.logging.error(f"Unexpected error during token refresh: {e}")
            return False
    
    def _update_env_file(self, new_token: str) -> bool:
        """
        Update .env file with new token.
        
        Args:
            new_token: The new JWT token to save
            
        Returns:
            True if update succeeded, False otherwise
        """
        env_path = '.env'
        
        try:
            # Read existing .env
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    lines = f.readlines()
            else:
                logger.warning(f"{env_path} does not exist, creating new file")
                lines = []
            
            # Update or add wallet-specific JWT variable (e.g. SUPABASE_JWT_5FEtvBzsh)
            wallet_updated = False
            for i, line in enumerate(lines):
                if line.strip().startswith(f'{self.env_var_name}='):
                    lines[i] = f'{self.env_var_name}={new_token}\n'
                    wallet_updated = True
                    break
            
            if not wallet_updated:
                # Add new line if not found
                if lines and not lines[-1].endswith('\n'):
                    lines.append('\n')
                lines.append(f'{self.env_var_name}={new_token}\n')
            
            # Also update generic SUPABASE_JWT for backward compatibility
            generic_updated = False
            for i, line in enumerate(lines):
                if line.strip().startswith('SUPABASE_JWT=') and not line.strip().startswith('SUPABASE_JWT_'):
                    lines[i] = f'SUPABASE_JWT={new_token}\n'
                    generic_updated = True
                    break
            
            if not generic_updated:
                # Add new line if not found
                if lines and not lines[-1].endswith('\n'):
                    lines.append('\n')
                lines.append(f'SUPABASE_JWT={new_token}\n')
            
            # Write back to .env
            with open(env_path, 'w') as f:
                f.writelines(lines)
            
            logger.info(f"Updated {env_path} with new token")
            return True
            
        except IOError as e:
            logger.error(f"Error updating .env file: {e}")
            return False
    
    def refresh_if_needed(self, threshold_hours: int = 1) -> bool:
        """
        Check and refresh token if needed.
        
        Args:
            threshold_hours: Refresh if token expires within this many hours
            
        Returns:
            True if token is valid (either didn't need refresh or refresh succeeded)
        """
        if not self.needs_refresh(threshold_hours):
            return True
        
        if self.is_expired():
            logger.warning("⚠️ Token has already expired! Attempting immediate refresh...")
        else:
            time_remaining = (self.token_expires - datetime.now()).total_seconds() / 3600
            logger.info(f"🔄 Token expiring in {time_remaining:.1f} hours, refreshing...")
        
        return self.refresh_token()
    
    def get_token(self) -> Optional[str]:
        """
        Get current valid token, refreshing if necessary.
        
        Returns:
            Current JWT token or None if unavailable
        """
        if self.refresh_if_needed():
            return self.token
        return None
    
    def get_status(self) -> dict:
        """
        Get current token status information.
        
        Returns:
            Dictionary with token status details
        """
        if not self.token_expires:
            return {
                'valid': False,
                'error': 'No token or unable to parse'
            }
        
        now = datetime.now()
        time_remaining = (self.token_expires - now).total_seconds()
        
        return {
            'valid': time_remaining > 0,
            'role': self.role,
            'expires_at': self.token_expires.isoformat(),
            'hours_remaining': time_remaining / 3600,
            'needs_refresh': self.needs_refresh(),
            'is_expired': self.is_expired()
        }

    def request_new_token(self) -> Optional[str]:
        """
        Request a new JWT token from the issuance endpoint
        Uses cryptographic signature to prove ownership of the hotkey
        
        Returns:
            str: New JWT token if successful, None otherwise
        """
        try:
            if not self.wallet:
                print("❌ Wallet not provided - cannot sign request")
                bt.logging.error("❌ Wallet not provided - cannot sign request")
                return None
            
            # Create timestamped message
            timestamp = int(time.time())
            message = f"leadpoet-jwt-request:{timestamp}"
            
            # Sign message with hotkey private key
            print("🔐 Signing request with hotkey...")
            bt.logging.info("🔐 Signing request with hotkey...")
            signature = self.wallet.hotkey.sign(message.encode())
            
            # Validate required parameters
            if not self.netuid or not self.network:
                print("❌ Missing netuid or network - cannot request token")
                bt.logging.error("❌ Missing netuid or network - cannot request token")
                return None
            
            # Prepare request payload
            payload = {
                "hotkey": self.hotkey,
                "message": message,
                "signature": signature.hex(),
                "timestamp": timestamp,
                "netuid": self.netuid,
                "network": self.network
            }
            
            # Include Supabase anon key for Edge Function authentication
            anon_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFwbHdvaXNscGxrY2VndmRtYmltIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ4NDcwMDUsImV4cCI6MjA2MDQyMzAwNX0.5E0WjAthYDXaCWY6qjzXm2k20EhadWfigak9hleKZk8"
            
            print(f"📤 Requesting new JWT token for {self.hotkey[:10]}...")
            bt.logging.info(f"📤 Requesting new JWT token for {self.hotkey[:10]}...")
            response = requests.post(
                self.token_endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {anon_key}",  # Required for Supabase API Gateway
                    "apikey": anon_key,  # Also required for Edge Function routing
                    "Content-Type": "application/json"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                new_token = data.get("token")
                role = data.get("role")
                uid = data.get("uid")
                
                print(f"✅ New token received - Role: {role}, UID: {uid}")
                bt.logging.info(f"✅ New token received - Role: {role}, UID: {uid}")
                return new_token
            
            elif response.status_code == 429:
                print("⚠️ Rate limit exceeded. Try again later.")
                bt.logging.warning("⚠️ Rate limit exceeded. Try again later.")
                return None
            
            elif response.status_code == 401:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", "Unauthorized")
                    print(f"❌ Authentication failed: {error_msg}")
                    print(f"   Full response: {error_data}")
                    bt.logging.error(f"❌ Authentication failed: {error_msg}")
                    bt.logging.error(f"   Full response: {error_data}")
                except Exception:
                    print(f"❌ Authentication failed: {response.text[:300]}")
                    bt.logging.error(f"❌ Authentication failed: {response.text[:300]}")
                return None
            
            elif response.status_code == 403:
                error_msg = response.json().get("error", "Forbidden")
                print(f"❌ Access denied: {error_msg}")
                bt.logging.error(f"❌ Access denied: {error_msg}")
                return None
            
            else:
                print(f"❌ Token request failed: {response.status_code} - {response.text}")
                bt.logging.error(f"❌ Token request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error requesting token: {e}")
            bt.logging.error(f"❌ Error requesting token: {e}")
            import traceback
            traceback.print_exc()
            return None


# Example standalone usage
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get hotkey from command line or environment
    hotkey = sys.argv[1] if len(sys.argv) > 1 else os.getenv('HOTKEY')
    
    if not hotkey:
        print("Usage: python token_manager.py <hotkey>")
        print("Or set HOTKEY environment variable")
        sys.exit(1)
    
    # Create token manager
    manager = TokenManager(hotkey=hotkey)
    
    # Check status
    status = manager.get_status()
    print("\n📊 Token Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Refresh if needed
    if manager.refresh_if_needed():
        print("\n✅ Token is valid and ready to use")
    else:
        print("\n❌ Token refresh failed")
        sys.exit(1)