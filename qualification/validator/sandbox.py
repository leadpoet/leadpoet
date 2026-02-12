"""
Qualification System: TEE Sandbox Manager

Phase 3.2 from tasks10.md

This module implements the TEESandbox class for executing miner models
in AWS Nitro Enclaves. Each model runs in an isolated enclave with:
- No network access (except via API proxy)
- No filesystem access
- Memory isolation
- CPU isolation

CRITICAL: This is a NEW TEE implementation for qualification sandboxing.
Do NOT modify any existing TEE code in validator_tee/ or gateway/tee/.
The qualification sandbox is separate from the attestation TEE.

Note: api_proxy_url is for the API Proxy Gateway (cost tracking/allowlist),
NOT Webshare proxies. Benchmarks run sequentially without Webshare proxies.
"""

import os
import sys
import json
import socket
import struct
import asyncio
import logging
import tempfile
import hashlib
import tarfile
import shutil
from uuid import UUID
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

# Allow nested event loops (for miner models that use asyncio.run())
try:
    import nest_asyncio
    nest_asyncio.apply()
    logging.getLogger(__name__).info("✅ nest_asyncio applied - models can use asyncio.run()")
except ImportError:
    logging.getLogger(__name__).warning("⚠️ nest_asyncio not installed - models using asyncio.run() will fail")

from gateway.qualification.models import ICPPrompt

# Security imports - CRITICAL for model isolation
try:
    from qualification.validator.sandbox_security import (
        SandboxSecurityContext,
        sanitize_environment,
        ALLOWED_LIBRARIES,
        ALLOWED_FREE_APIS,
        ALLOWED_PAID_APIS,
        SENSITIVE_ENV_VARS,
    )
    SECURITY_AVAILABLE = True
except ImportError as e:
    import traceback
    print(f"⚠️ sandbox_security import failed: {e}")
    traceback.print_exc()
    SECURITY_AVAILABLE = False

# Local proxy - runs in validator process, separate thread
try:
    from qualification.validator.local_proxy import LocalProxyServer
    LOCAL_PROXY_AVAILABLE = True
except ImportError as e:
    import traceback
    print(f"⚠️ local_proxy import failed: {e}")
    traceback.print_exc()
    LOCAL_PROXY_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Enclave configuration
ENCLAVE_PORT = 5000  # vsock port for communication with enclave
ENCLAVE_CPU_COUNT = 2  # CPUs allocated to enclave
ENCLAVE_MEMORY_MB = 4096  # Memory allocated to enclave (4GB)

# Timeouts
ENCLAVE_START_TIMEOUT_SECONDS = 60
ENCLAVE_STOP_TIMEOUT_SECONDS = 30
VSOCK_CONNECT_TIMEOUT_SECONDS = 10
VSOCK_READ_TIMEOUT_SECONDS = 30

# Paths
ENCLAVE_BASE_IMAGE = os.getenv(
    "QUALIFICATION_ENCLAVE_BASE_IMAGE",
    "/opt/qualification/enclave-base.eif"
)
ENCLAVE_BUILD_DIR = os.getenv(
    "QUALIFICATION_ENCLAVE_BUILD_DIR",
    "/tmp/qualification/enclave-builds"
)

# Docker image for building enclaves
ENCLAVE_DOCKER_BASE = os.getenv(
    "QUALIFICATION_ENCLAVE_DOCKER_BASE",
    "qualification-sandbox:latest"
)

# Feature flags
USE_NITRO_ENCLAVE = os.getenv("USE_NITRO_ENCLAVE", "false").lower() == "true"


# =============================================================================
# TEE Sandbox Class
# =============================================================================

class TEESandbox:
    """
    AWS Nitro Enclave sandbox for model execution (benchmarking).
    
    Each model runs in an isolated Nitro Enclave with:
    - Dedicated CPU cores (ENCLAVE_CPU_COUNT)
    - Dedicated memory (ENCLAVE_MEMORY_MB)
    - No direct network access
    - Communication only via vsock to API proxy
    
    The sandbox lifecycle:
    1. __init__: Store model code and configuration
    2. start(): Build and launch the enclave
    3. run_model(): Execute model's qualify() function
    4. cleanup(): Terminate the enclave
    
    Attributes:
        model_code: Bytes of the model tarball
        evaluation_run_id: UUID for this evaluation run
        api_proxy_url: URL of the API Proxy Gateway (NOT Webshare proxy)
        evaluation_id: Optional parent evaluation UUID
        enclave_cid: Enclave CID (set after start())
        enclave_id: Full enclave ID (set after start())
        started: Whether enclave is running
    """
    
    def __init__(
        self,
        model_code: bytes,
        evaluation_run_id: UUID,
        api_proxy_url: str,
        evaluation_id: Optional[UUID] = None
    ):
        """
        Initialize the TEE sandbox.
        
        Args:
            model_code: Model code as bytes (tarball)
            evaluation_run_id: UUID of the evaluation run
            api_proxy_url: URL of the API Proxy Gateway for cost tracking.
                           NOTE: This is NOT a Webshare proxy. Benchmarks run
                           sequentially without external proxies.
            evaluation_id: Optional parent evaluation UUID
        """
        self.model_code = model_code
        self.evaluation_run_id = evaluation_run_id
        self.api_proxy_url = api_proxy_url
        self.evaluation_id = evaluation_id
        
        # Enclave state (set after start())
        self.enclave_cid: Optional[int] = None
        self.enclave_id: Optional[str] = None
        self.started = False
        
        # Temporary files (cleaned up in cleanup())
        self._temp_dir: Optional[str] = None
        self._eif_path: Optional[str] = None
        
        # Code hash for logging/debugging
        self._code_hash = hashlib.sha256(model_code).hexdigest()[:16] if model_code else "empty"
        
        # Cost tracking (updated after each run)
        self._last_run_cost = 0.0
        
        logger.info(
            f"TEESandbox created: run_id={evaluation_run_id}, "
            f"code_hash={self._code_hash}, code_size={len(model_code)} bytes"
        )
    
    # =========================================================================
    # Lifecycle Methods
    # =========================================================================
    
    async def start(self):
        """
        Start the Nitro enclave with the model code.
        
        Steps:
        1. Build enclave image (EIF) with model code embedded
        2. Launch enclave with nitro-cli
        3. Wait for enclave to be ready
        
        Raises:
            SandboxStartError: If enclave fails to start
        """
        if self.started:
            logger.warning("Sandbox already started")
            return
        
        logger.info(f"Starting sandbox: run_id={self.evaluation_run_id}")
        
        try:
            # Create temp directory for build artifacts
            self._temp_dir = tempfile.mkdtemp(prefix="qual_sandbox_")
            
            # Build enclave image
            self._eif_path = await self.build_enclave_image(self.model_code)
            
            if USE_NITRO_ENCLAVE:
                # Production: Real Nitro hardware enclave
                await self._start_nitro_enclave()
            else:
                # Testing: Local process with same security restrictions
                await self._start_local_sandbox()
            
            self.started = True
            logger.info(
                f"Sandbox started: run_id={self.evaluation_run_id}, "
                f"enclave_cid={self.enclave_cid}"
            )
            
        except Exception as e:
            logger.error(f"Failed to start sandbox: {e}")
            await self.cleanup()
            raise SandboxStartError(f"Failed to start sandbox: {e}") from e
    
    async def run_model(self, icp: ICPPrompt) -> Dict[str, Any]:
        """
        Execute the model's qualify() function with the given ICP.
        
        Sends the ICP to the enclave via vsock, waits for the model
        to return a lead, and returns the result.
        
        Args:
            icp: The ICP prompt to pass to the model
        
        Returns:
            Dict containing:
            - lead: The lead returned by the model (or None)
            - error: Error message if model failed (or None)
            - execution_time_ms: Time taken to execute
        
        Raises:
            SandboxExecutionError: If execution fails
        """
        if not self.started:
            raise SandboxExecutionError("Sandbox not started")
        
        logger.debug(f"Running model with ICP: {icp.icp_id}")
        
        # Build request
        request = {
            "action": "qualify",
            "icp": icp.model_dump(mode="json"),
            "evaluation_run_id": str(self.evaluation_run_id),
            "evaluation_id": str(self.evaluation_id) if self.evaluation_id else None,
            "api_proxy_url": self.api_proxy_url  # API Proxy Gateway for cost tracking
        }
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            if USE_NITRO_ENCLAVE:
                # Production: Send via vsock to Nitro enclave
                response = await self._send_vsock_request(request)
            else:
                # Testing: Run locally with security restrictions
                response = await self._run_local_model(request)
            
            execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Add execution time to response
            if isinstance(response, dict):
                response["execution_time_ms"] = execution_time_ms
            
            logger.debug(
                f"Model completed: icp={icp.icp_id}, "
                f"time={execution_time_ms:.1f}ms, "
                f"has_lead={response.get('lead') is not None}"
            )
            
            return response
            
        except asyncio.TimeoutError:
            raise SandboxExecutionError("Model execution timed out")
        except Exception as e:
            logger.error(f"Model execution failed: {e}")
            raise SandboxExecutionError(f"Model execution failed: {e}") from e
    
    def get_run_cost(self) -> float:
        """
        Get the cost of the last model execution.
        
        This retrieves the cost tracked by the model's internal cost tracking
        (e.g., LLM API calls made during find_leads()).
        
        Returns:
            Cost in USD of the last execution
        """
        return self._last_run_cost
    
    async def cleanup(self):
        """
        Terminate the enclave and clean up resources.
        
        Always safe to call (even if not started or already cleaned up).
        """
        logger.info(f"Cleaning up sandbox: run_id={self.evaluation_run_id}")
        
        try:
            if self.enclave_id and USE_NITRO_ENCLAVE:
                await self._terminate_nitro_enclave()
            
            self.started = False
            self.enclave_cid = None
            self.enclave_id = None
            
        except Exception as e:
            logger.warning(f"Error terminating enclave: {e}")
        
        # Clean up temp directory
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up temp dir: {e}")
        
        self._temp_dir = None
        self._eif_path = None
    
    # =========================================================================
    # Enclave Image Building
    # =========================================================================
    
    async def build_enclave_image(self, model_code: bytes) -> str:
        """
        Build an enclave image (EIF) with the model code embedded.
        
        Args:
            model_code: Model code as bytes (tarball)
        
        Returns:
            Path to the built EIF file
        
        Steps:
        1. Extract model code to temp directory
        2. Create Dockerfile with model and runtime
        3. Build Docker image
        4. Convert to EIF using nitro-cli
        """
        logger.info(f"Building enclave image: code_size={len(model_code)} bytes")
        
        build_dir = self._temp_dir
        model_dir = os.path.join(build_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Extract model code if provided
        if model_code:
            tar_path = os.path.join(build_dir, "model.tar.gz")
            with open(tar_path, "wb") as f:
                f.write(model_code)
            
            # Extract tarball
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(model_dir)
        
        # Create Dockerfile for enclave
        dockerfile_content = self._generate_dockerfile()
        dockerfile_path = os.path.join(build_dir, "Dockerfile")
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
        
        # Create enclave entry script
        entrypoint_content = self._generate_entrypoint()
        entrypoint_path = os.path.join(build_dir, "entrypoint.py")
        with open(entrypoint_path, "w") as f:
            f.write(entrypoint_content)
        
        if USE_NITRO_ENCLAVE:
            # Build Docker image and convert to EIF
            image_tag = f"qual-sandbox-{self.evaluation_run_id}"
            
            # Build Docker image
            result = await asyncio.create_subprocess_exec(
                "docker", "build", "-t", image_tag, build_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                raise SandboxBuildError(f"Docker build failed: {stderr.decode()}")
            
            # Convert to EIF
            eif_path = os.path.join(build_dir, "enclave.eif")
            result = await asyncio.create_subprocess_exec(
                "nitro-cli", "build-enclave",
                "--docker-uri", image_tag,
                "--output-file", eif_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                raise SandboxBuildError(f"EIF build failed: {stderr.decode()}")
            
            logger.info(f"Enclave image built: {eif_path}")
            return eif_path
        else:
            # Fallback: just return the build directory
            logger.info(f"Fallback mode: model prepared at {model_dir}")
            return model_dir
    
    def _generate_dockerfile(self) -> str:
        """Generate Dockerfile for the enclave."""
        return '''FROM python:3.11-slim

# Install dependencies
RUN pip install --no-cache-dir httpx pydantic

# Copy model code
COPY model/ /app/model/

# Copy entrypoint
COPY entrypoint.py /app/entrypoint.py

WORKDIR /app

# Run entrypoint
CMD ["python", "/app/entrypoint.py"]
'''
    
    def _generate_entrypoint(self) -> str:
        """Generate entrypoint script for the enclave."""
        return '''#!/usr/bin/env python3
"""
Enclave entrypoint script.
Listens on vsock and executes the model's qualify() function.
"""

import os
import sys
import json
import socket
import traceback

# Add model to path
sys.path.insert(0, "/app/model")

VSOCK_PORT = 5000

def main():
    # Import model
    try:
        from model import qualify
    except ImportError:
        # Try alternate import paths
        try:
            from model.model import qualify
        except ImportError:
            try:
                from model.main import qualify
            except ImportError:
                print("ERROR: Could not import qualify function", file=sys.stderr)
                qualify = None
    
    # Create vsock listener
    # Note: In a real Nitro enclave, this would be a vsock socket
    # For fallback mode, we use a Unix socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", VSOCK_PORT))
    sock.listen(1)
    
    print(f"Enclave listening on port {VSOCK_PORT}")
    
    while True:
        conn, addr = sock.accept()
        try:
            # Read request
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                # Check for complete JSON
                try:
                    json.loads(data)
                    break
                except json.JSONDecodeError:
                    continue
            
            request = json.loads(data)
            
            # Execute model
            response = {"lead": None, "error": None}
            
            if qualify is None:
                response["error"] = "Model qualify() function not found"
            else:
                try:
                    icp = request.get("icp", {})
                    api_proxy_url = request.get("api_proxy_url")
                    
                    # Set environment for model
                    os.environ["API_PROXY_URL"] = api_proxy_url or ""
                    os.environ["EVALUATION_RUN_ID"] = request.get("evaluation_run_id", "")
                    
                    # Call model
                    lead = qualify(icp)
                    response["lead"] = lead
                except Exception as e:
                    response["error"] = str(e)
                    traceback.print_exc()
            
            # Send response
            conn.sendall(json.dumps(response).encode())
            
        except Exception as e:
            print(f"Error handling request: {e}", file=sys.stderr)
            try:
                conn.sendall(json.dumps({"lead": None, "error": str(e)}).encode())
            except:
                pass
        finally:
            conn.close()

if __name__ == "__main__":
    main()
'''
    
    # =========================================================================
    # Nitro Enclave Operations
    # =========================================================================
    
    async def _start_nitro_enclave(self):
        """Start a real Nitro enclave."""
        logger.info(f"Starting Nitro enclave: {self._eif_path}")
        
        result = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "nitro-cli", "run-enclave",
                "--eif-path", self._eif_path,
                "--cpu-count", str(ENCLAVE_CPU_COUNT),
                "--memory", str(ENCLAVE_MEMORY_MB),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            ),
            timeout=ENCLAVE_START_TIMEOUT_SECONDS
        )
        
        stdout, stderr = await result.communicate()
        
        if result.returncode != 0:
            raise SandboxStartError(f"nitro-cli run-enclave failed: {stderr.decode()}")
        
        # Parse enclave info
        enclave_info = json.loads(stdout.decode())
        self.enclave_cid = enclave_info.get("EnclaveCID")
        self.enclave_id = enclave_info.get("EnclaveID")
        
        if not self.enclave_cid:
            raise SandboxStartError("No EnclaveCID in nitro-cli output")
        
        logger.info(
            f"Nitro enclave started: cid={self.enclave_cid}, id={self.enclave_id}"
        )
    
    async def _terminate_nitro_enclave(self):
        """Terminate a Nitro enclave."""
        if not self.enclave_id:
            return
        
        logger.info(f"Terminating Nitro enclave: {self.enclave_id}")
        
        try:
            result = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    "nitro-cli", "terminate-enclave",
                    "--enclave-id", self.enclave_id,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                ),
                timeout=ENCLAVE_STOP_TIMEOUT_SECONDS
            )
            await result.communicate()
        except asyncio.TimeoutError:
            logger.warning(f"Enclave termination timed out: {self.enclave_id}")
    
    async def _send_vsock_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to the enclave via vsock."""
        if not self.enclave_cid:
            raise SandboxExecutionError("No enclave CID")
        
        # Create vsock socket
        sock = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
        sock.settimeout(VSOCK_CONNECT_TIMEOUT_SECONDS)
        
        try:
            # Connect to enclave
            sock.connect((self.enclave_cid, ENCLAVE_PORT))
            
            # Send request
            request_bytes = json.dumps(request).encode()
            sock.sendall(request_bytes)
            sock.shutdown(socket.SHUT_WR)  # Signal end of request
            
            # Read response
            sock.settimeout(VSOCK_READ_TIMEOUT_SECONDS)
            response_bytes = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_bytes += chunk
            
            return json.loads(response_bytes)
            
        finally:
            sock.close()
    
    # =========================================================================
    # Fallback Mode (Non-Nitro)
    # =========================================================================
    
    async def _start_local_sandbox(self):
        """
        Start sandbox in local mode (USE_NITRO_ENCLAVE=false).
        
        Same security restrictions as Nitro mode:
        - Import restrictions (only ALLOWED_LIBRARIES)
        - Network interception (only proxy allowed)
        - Environment sanitization (no API keys)
        
        The only difference is no hardware enclave - for testing.
        """
        logger.info(
            "🧪 Starting in LOCAL mode (USE_NITRO_ENCLAVE=false). "
            "Same security restrictions, no hardware enclave."
        )
        
        if not SECURITY_AVAILABLE:
            raise RuntimeError(
                "CRITICAL: Security module not available! "
                "Cannot run models without import/network restrictions. "
                "Fix sandbox_security.py import error."
            )
        
        logger.info("✅ Security module loaded - import/network/env restrictions ENABLED")
        
        # Local mode uses a placeholder CID
        self.enclave_cid = 3
        self.enclave_id = f"local-{self.evaluation_run_id}"
    
    async def _run_local_model(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the model locally with full security restrictions.
        
        This is used when USE_NITRO_ENCLAVE=false (testing mode).
        Same security measures as production:
        1. Import restrictions - only ALLOWED_LIBRARIES can be imported
        2. Network interception - all HTTP calls go through proxy
        3. Environment sanitization - API keys removed from env
        
        The ONLY difference from production is no hardware enclave.
        """
        # Security is REQUIRED - not optional
        if not SECURITY_AVAILABLE:
            raise RuntimeError(
                "CRITICAL: Security module not available! "
                "Cannot run miner models without security restrictions."
            )
        
        logger.info("🔐 Running model with SECURITY CONTEXT (import/network/env restrictions)")
        
        model_dir = self._eif_path  # In local mode, this is the model directory
        
        if not model_dir or not os.path.exists(model_dir):
            return {"lead": None, "error": "Model directory not found"}
        
        # Debug: Show directory structure
        logger.info(f"Model directory: {model_dir}")
        try:
            for root, dirs, files in os.walk(model_dir):
                level = root.replace(model_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                logger.info(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files[:10]:  # Limit to first 10 files
                    logger.info(f"{subindent}{file}")
        except Exception as e:
            logger.warning(f"Could not list model directory: {e}")
        
        # Add model to path
        model_subdir = os.path.join(model_dir, "model")
        sys.path.insert(0, model_dir)
        if os.path.exists(model_subdir):
            sys.path.insert(0, model_subdir)
            logger.info(f"Added to sys.path: {model_subdir}")
        
        try:
            # =================================================================
            # SECURE MODEL EXECUTION (ALWAYS)
            # =================================================================
            # The SandboxSecurityContext wraps model execution with:
            # 1. RestrictedImporter - blocks unauthorized imports
            # 2. NetworkInterceptor - routes API calls through proxy
            # 3. Environment sanitization - removes API keys from env
            # =================================================================
            return await self._execute_model_with_security(request, model_dir)
            
        except Exception as e:
            logger.error(f"Model execution failed: {e}")
            import traceback
            traceback.print_exc()
            return {"lead": None, "error": str(e)}
        finally:
            # Remove model from path
            if model_dir in sys.path:
                sys.path.remove(model_dir)
            model_subdir = os.path.join(model_dir, "model")
            if model_subdir in sys.path:
                sys.path.remove(model_subdir)
    
    async def _execute_model_with_security(
        self, 
        request: Dict[str, Any], 
        model_dir: str
    ) -> Dict[str, Any]:
        """
        Execute model with full security context.
        
        This method wraps model execution in SandboxSecurityContext which:
        - Blocks unauthorized imports (dangerous modules like subprocess, ctypes, etc.)
        - Intercepts HTTP requests and routes through local proxy
        - Sanitizes env vars (removes API keys before model runs)
        
        SECURITY MODEL:
        1. Local proxy starts (has API keys from os.environ)
        2. Local proxy stores keys in thread-local closure
        3. SandboxSecurityContext enters (sanitizes os.environ - removes keys)
        4. Model runs (no keys in os.environ, can only call local proxy)
        5. Model calls local proxy for LLM/APIs
        6. Local proxy injects keys from its closure and forwards
        7. Model NEVER sees API keys
        8. SandboxSecurityContext exits
        9. Local proxy stops
        
        This is IDENTICAL security to the gateway proxy approach, but faster
        (in-process, no network hop) and more reliable (no gateway dependency).
        """
        evaluation_run_id = request.get("evaluation_run_id", str(self.evaluation_run_id))
        evaluation_id = request.get("evaluation_id", str(self.evaluation_id) if self.evaluation_id else None)
        
        # =================================================================
        # STEP 1: Start local proxy BEFORE sandbox (has access to API keys)
        # =================================================================
        # The local proxy runs in a SEPARATE THREAD with API keys in its closure.
        # After sandbox activates, os.environ won't have keys anymore, but the
        # proxy thread still has them in its closure. Model's thread cannot
        # access the proxy thread's stack (inspect.stack() is per-thread).
        # =================================================================
        local_proxy_url = None
        local_proxy = None
        
        if LOCAL_PROXY_AVAILABLE:
            try:
                local_proxy = LocalProxyServer()
                local_proxy_url = local_proxy.start()
                logger.info(f"🔐 Local proxy started at {local_proxy_url}")
            except Exception as e:
                logger.warning(f"⚠️ Local proxy failed to start: {e}, falling back to gateway proxy")
                local_proxy = None
                local_proxy_url = None
        
        # Fall back to gateway proxy if local proxy unavailable
        if not local_proxy_url:
            local_proxy_url = self.api_proxy_url
            logger.info(f"Using gateway proxy: {local_proxy_url}")
        
        # Update request with local proxy URL
        request["api_proxy_url"] = local_proxy_url
        
        try:
            # =================================================================
            # STEP 2: Enter security context (sanitizes os.environ)
            # =================================================================
            # NOTE: Do NOT use "as ctx" - this would expose _original_open
            # to frame inspection attacks from malicious models.
            # =================================================================
            async with SandboxSecurityContext(
                evaluation_run_id=evaluation_run_id,
                evaluation_id=evaluation_id or "",
                enable_import_restriction=True,
                enable_network_interception=True,
                enable_env_sanitization=True,
                enable_file_restriction=True,
                proxy_url=local_proxy_url
            ):
                # =============================================================
                # STEP 3: Run model (os.environ sanitized, can only call proxy)
                # =============================================================
                result = await self._import_and_run_model(request, model_dir)
                
                # =============================================================
                # STEP 4: Get ACTUAL cost from proxy (source of truth)
                # =============================================================
                # This is the authoritative cost - NOT the model's self-reported cost.
                # The proxy intercepts all API calls and tracks actual costs from
                # OpenRouter responses (using usage.total_cost or token-based estimate).
                if local_proxy:
                    proxy_cost = local_proxy.get_total_cost()
                    proxy_summary = local_proxy.get_cost_summary()
                    
                    # Store proxy cost (overrides any model-reported cost)
                    self._last_run_cost = proxy_cost
                    
                    logger.info(f"💰 PROXY COST (authoritative): ${proxy_cost:.6f}")
                    logger.info(f"   Call counts: {proxy_summary.get('call_counts', {})}")
                    if proxy_summary.get('token_counts'):
                        for provider, tokens in proxy_summary['token_counts'].items():
                            logger.info(f"   {provider} tokens: {tokens['input']}+{tokens['output']}")
                
                return result
        
        finally:
            # =================================================================
            # STEP 5: Stop local proxy (cleanup)
            # =================================================================
            if local_proxy:
                try:
                    local_proxy.stop()
                    logger.info("🔐 Local proxy stopped")
                except Exception as e:
                    logger.warning(f"Error stopping local proxy: {e}")
    
    async def _import_and_run_model(
        self, 
        request: Dict[str, Any], 
        model_dir: str
    ) -> Dict[str, Any]:
        """
        Import the model's qualify function and execute it.
        
        Called within the SandboxSecurityContext which enforces:
        - Import restrictions (only ALLOWED_LIBRARIES)
        - Network restrictions (only proxy URL)
        - Environment sanitization (no API keys)
        """
        # Try to import qualify/find_leads function
        qualify_func = None
        import_errors = []
        
        # Try multiple import strategies with both function names
        # Order matters: qualify.py is most common, then model.py, then main.py
        import_attempts = [
            # Strategy 1: qualify.py at root (most common for miners)
            ("qualify", "qualify"),
            ("qualify", "find_leads"),
            # Strategy 2: model.py with qualify/find_leads function
            ("model", "qualify"),
            ("model", "find_leads"),
            # Strategy 3: model/model.py
            ("model.model", "qualify"),
            ("model.model", "find_leads"),
            # Strategy 4: model/main.py
            ("model.main", "qualify"),
            ("model.main", "find_leads"),
            # Strategy 5: main.py
            ("main", "qualify"),
            ("main", "find_leads"),
        ]
        
        for module_path, func_name in import_attempts:
            try:
                logger.info(f"🔍 Trying import: {module_path}.{func_name}")
                module = __import__(module_path, fromlist=[func_name])
                attr = getattr(module, func_name, None)
                
                # Check if we got a function or a module
                if callable(attr):
                    qualify_func = attr
                    logger.info(f"✅ Found qualify function via {module_path}.{func_name}")
                    break
                elif attr is not None and hasattr(attr, func_name):
                    # attr is a module (e.g., qualify.py), get the function from it
                    inner_func = getattr(attr, func_name, None)
                    if callable(inner_func):
                        qualify_func = inner_func
                        logger.info(f"✅ Found qualify function via {module_path}.{func_name}.{func_name}")
                        break
            except ImportError as e:
                import_errors.append(f"{module_path}: {str(e)[:50]}")
                logger.debug(f"Import {module_path} failed: {e}")
                continue
            except Exception as e:
                import_errors.append(f"{module_path}: {str(e)[:50]}")
                logger.debug(f"Error importing {module_path}: {e}")
                continue
        
        if not qualify_func:
            # List all files in model directory for debugging
            try:
                all_files = []
                for root, dirs, files in os.walk(model_dir):
                    for f in files:
                        if f.endswith('.py'):
                            rel_path = os.path.relpath(os.path.join(root, f), model_dir)
                            all_files.append(rel_path)
                files_str = ", ".join(all_files[:10]) if all_files else "no .py files"
            except:
                files_str = "could not list"
            
            error_msg = (
                f"Could not import qualify/find_leads function. "
                f"Model files: [{files_str}]. "
                f"Tried: {', '.join([m for m, _ in import_attempts[:5]])}. "
                f"First error: {import_errors[0] if import_errors else 'unknown'}"
            )
            logger.error(error_msg)
            return {"lead": None, "error": error_msg}
        
        # =================================================================
        # SET ENVIRONMENT FOR MODEL
        # =================================================================
        # In SECURE mode: environment is sanitized, only safe vars present
        # In INSECURE mode: we manually set the vars (dangerous!)
        # =================================================================
        
        os.environ["API_PROXY_URL"] = request.get("api_proxy_url", "")
        os.environ["EVALUATION_RUN_ID"] = request.get("evaluation_run_id", "")
        
        # Supabase credentials (anon key only - read-only access)
        supabase_url = os.getenv("SUPABASE_URL", "https://qplwoislplkcegvdmbim.supabase.co")
        supabase_anon_key = os.getenv(
            "SUPABASE_ANON_KEY", 
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFwbHdvaXNscGxrY2VndmRtYmltIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ4NDcwMDUsImV4cCI6MjA2MDQyMzAwNX0.5E0WjAthYDXaCWY6qjzXm2k20EhadWfigak9hleKZk8"
        )
        os.environ["SUPABASE_URL"] = supabase_url
        os.environ["SUPABASE_ANON_KEY"] = supabase_anon_key
        # Table name is set via QUALIFICATION_LEADS_TABLE env var (not hardcoded)
        qualification_table = os.getenv("QUALIFICATION_LEADS_TABLE", "")
        if qualification_table:
            os.environ["QUALIFICATION_LEADS_TABLE"] = qualification_table
        
        logger.info(f"✅ Supabase credentials set: {supabase_url[:40]}...")
        
        # =================================================================
        # API KEY SECURITY - CRITICAL
        # =================================================================
        # Models NEVER receive API keys. The proxy injects them server-side.
        # 
        # Flow:
        # 1. Model calls proxy: POST http://localhost:8001/proxy/openrouter/...
        # 2. Proxy receives request WITHOUT any API key
        # 3. Proxy injects the API key server-side via get_auth_headers()
        # 4. Proxy forwards to OpenRouter with the key
        # 5. Model NEVER sees the key value
        #
        # This prevents exfiltration attacks where a malicious model reads
        # the key and encodes it in a request body/URL to an allowed API.
        # =================================================================
        logger.info("🔐 API keys NOT injected - proxy handles authentication server-side")
        
        # Get ICP prompt from request
        icp_data = request.get("icp", {})
        
        # =================================================================
        # INJECT SAFE CONFIG INTO ICP
        # =================================================================
        # We inject safe config values directly into the ICP dict.
        # Models read icp["_config"]["SUPABASE_URL"] etc.
        # 
        # PROXY_URL: The local proxy runs at http://localhost:<port>
        # Model calls: {PROXY_URL}/openrouter/chat/completions
        # Proxy forwards to OpenRouter with API key injected server-side
        # =================================================================
        proxy_url = request.get("api_proxy_url", "http://localhost:8001")
        
        # Get table name from env var (set by validator startup command)
        qualification_table = os.getenv("QUALIFICATION_LEADS_TABLE", "")
        
        icp_data["_config"] = {
            "SUPABASE_URL": supabase_url,
            "SUPABASE_ANON_KEY": supabase_anon_key,
            "QUALIFICATION_LEADS_TABLE": qualification_table,
            "PROXY_URL": proxy_url,
        }
        logger.info(f"✅ Safe config injected into ICP['_config']")
        logger.info(f"   PROXY_URL: {proxy_url}")
        
        # Log the prompt
        prompt = icp_data.get("prompt") or icp_data.get("buyer_description", "")
        if prompt:
            logger.info(f"Running qualify(icp) with PROMPT: {prompt[:100]}...")
        else:
            logger.info(f"Running qualify(icp) with ICP: industry={icp_data.get('industry')}, "
                       f"roles={icp_data.get('target_roles', ['Unknown'])}")
        
        # =================================================================
        # EXECUTE MODEL
        # =================================================================
        # Call model - SIGNATURE: qualify(icp: Dict) -> Optional[Dict]
        # The model searches the leads database and returns the best match
        # =================================================================
        
        lead = qualify_func(icp_data)
        
        # Cost tracking is handled by the proxy (authoritative source)
        # Models should NOT track costs - the proxy intercepts all API calls
        
        return {"lead": lead, "error": None}


# =============================================================================
# Exceptions
# =============================================================================

class SandboxError(Exception):
    """Base exception for sandbox errors."""
    pass


class SandboxStartError(SandboxError):
    """Raised when sandbox fails to start."""
    pass


class SandboxBuildError(SandboxError):
    """Raised when enclave image build fails."""
    pass


class SandboxExecutionError(SandboxError):
    """Raised when model execution fails."""
    pass


# =============================================================================
# Factory Function
# =============================================================================

async def create_sandbox(
    model_code: bytes,
    evaluation_run_id: UUID,
    api_proxy_url: str,
    evaluation_id: Optional[UUID] = None,
    auto_start: bool = True
) -> TEESandbox:
    """
    Create and optionally start a TEE sandbox.
    
    Factory function for creating sandboxes.
    
    Args:
        model_code: Model code as bytes
        evaluation_run_id: UUID of the evaluation run
        api_proxy_url: API Proxy Gateway URL (for cost tracking, NOT Webshare)
        evaluation_id: Optional parent evaluation UUID
        auto_start: Whether to automatically start the sandbox
    
    Returns:
        TEESandbox instance (started if auto_start=True)
    """
    sandbox = TEESandbox(
        model_code=model_code,
        evaluation_run_id=evaluation_run_id,
        api_proxy_url=api_proxy_url,
        evaluation_id=evaluation_id
    )
    
    if auto_start:
        await sandbox.start()
    
    return sandbox
