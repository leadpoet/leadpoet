"""
Qualification System: Hardcoding Detection Module

This module analyzes submitted model code BEFORE execution to detect:
1. Hardcoded answers (lookup tables, pre-computed results)
2. Hard steering (gaming the evaluation with specific patterns)
3. Obviously malicious patterns (downloading payloads, accessing sensitive files)

HOW IT WORKS:
- Extracts ALL files from the submitted tarball
- ALL files count toward 200KB limit (py + md + txt + json + etc.)
- Only .py files are sent to o3-mini for analysis
- 200KB ≈ 50K tokens, well within o3-mini's 200K context

SIZE LIMIT: 200KB total for all files. Miner decides how to allocate:
- Big README? Less space for code.
- Multiple model files? Less space for docs.

Author: LeadPoet
"""

import os
import json
import tarfile
import tempfile
import logging
import httpx
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

def get_max_submission_size() -> int:
    """Get the max total submission size from config or use default (200KB)."""
    return _get_config_value("HARDCODING_MAX_SUBMISSION_SIZE_BYTES", 200_000)

# LLM Model for hardcoding detection - Claude Sonnet 4.5 (1M context window)
DETECTION_MODEL_ID = "anthropic/claude-sonnet-4.5"
DETECTION_MODEL_COST_INPUT = 1.10   # $ per 1M tokens
DETECTION_MODEL_COST_OUTPUT = 4.40  # $ per 1M tokens

# Default confidence threshold (can be overridden by config)
DEFAULT_REJECTION_THRESHOLD = 70  # 70% = likely hardcoded


def _get_config_value(attr: str, default: Any) -> Any:
    """Get config value with fallback to default."""
    try:
        from gateway.qualification.config import CONFIG
        return getattr(CONFIG, attr, default)
    except ImportError:
        return default


def get_rejection_threshold() -> int:
    """Get the configured rejection threshold."""
    return _get_config_value("HARDCODING_REJECTION_THRESHOLD", DEFAULT_REJECTION_THRESHOLD)


def is_detection_enabled() -> bool:
    """Check if hardcoding detection is enabled."""
    return _get_config_value("ENABLE_HARDCODING_DETECTION", True)


# =============================================================================
# Main Detection Function
# =============================================================================

async def analyze_model_for_hardcoding(
    model_code: bytes,
    icp_samples: List[Dict[str, Any]],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze a model submission for hardcoding patterns using an LLM.
    
    This runs BEFORE the model is executed in the sandbox. If the model
    appears to be hardcoded (confidence >= threshold), it should be rejected
    without running.
    
    SECURITY: The FULL code is sent to the LLM - no truncation. Models
    exceeding 100KB are rejected to ensure complete analysis.
    
    Args:
        model_code: Tarball bytes of the submitted model
        icp_samples: Sample ICPs that will be used for testing (3-5 examples)
        api_key: OpenRouter API key (defaults to env var)
    
    Returns:
        {
            "passed": bool,  # True if model appears legitimate, False if hardcoded
            "confidence_hardcoded": int,  # 0-100 (0=valid, 100=obviously hardcoded)
            "red_flags": List[str],  # Specific patterns found
            "evidence": str,  # LLM's explanation
            "model_used": str,  # Which LLM was used
            "analysis_cost_usd": float,  # Cost of this analysis
        }
    """
    # Check if detection is enabled
    if not is_detection_enabled():
        logger.info("Hardcoding detection is disabled via config")
        return {
            "passed": True,
            "confidence_hardcoded": 0,
            "red_flags": [],
            "evidence": "Hardcoding detection disabled",
            "model_used": None,
            "analysis_cost_usd": 0.0
        }
    
    try:
        # Get API key
        # SECURITY: Qualification uses SEPARATE API keys with limited funds.
        # TODO: After beta release, change back to "OPENROUTER_API_KEY" (shared with sourcing)
        openrouter_key = api_key or os.environ.get("QUALIFICATION_OPENROUTER_API_KEY")
        if not openrouter_key:
            logger.warning("No OpenRouter API key for hardcoding detection, skipping check")
            return {
                "passed": True,  # Allow if we can't check
                "confidence_hardcoded": 0,
                "red_flags": [],
                "evidence": "Skipped - no API key available",
                "model_used": None,
                "analysis_cost_usd": 0.0
            }
        
        # Extract code from tarball (ALL files count toward size, only .py analyzed)
        py_content, py_file_count, total_size = await _extract_code_from_tarball(model_code)
        
        # SECURITY: Check total size of ALL files FIRST (prevents memory attacks)
        max_size = get_max_submission_size()
        if total_size > max_size:
            logger.warning(
                f"❌ Model REJECTED: Total size {total_size:,} bytes exceeds "
                f"limit of {max_size:,} bytes (200KB)"
            )
            return {
                "passed": False,
                "confidence_hardcoded": 100,
                "red_flags": [f"Total submission size {total_size:,} bytes exceeds {max_size:,} byte limit"],
                "evidence": (
                    f"Model submission contains {total_size:,} bytes total across all files "
                    f"(py, md, txt, json, etc.). This exceeds the 200KB limit. "
                    f"Reduce file sizes or remove unnecessary files."
                ),
                "model_used": None,
                "analysis_cost_usd": 0.0
            }
        
        if not py_content:
            logger.warning("Could not extract Python code from tarball")
            return {
                "passed": True,  # Allow if no .py files (will fail elsewhere)
                "confidence_hardcoded": 0,
                "red_flags": [],
                "evidence": "No Python files found in tarball",
                "model_used": None,
                "analysis_cost_usd": 0.0
            }
        
        py_size = len(py_content.encode('utf-8'))
        logger.info(f"   Submission: {total_size:,} bytes total, {py_size:,} bytes Python across {py_file_count} file(s)")
        
        # Build the analysis prompt with FULL Python code and real ICP samples
        prompt = _build_analysis_prompt(py_content, icp_samples)
        
        # Get timeout from config
        timeout = _get_config_value("HARDCODING_DETECTION_TIMEOUT", 120)
        
        # Call the LLM (o3-mini)
        analysis_result, cost = await _call_reasoning_llm(
            prompt=prompt,
            api_key=openrouter_key,
            timeout=float(timeout)
        )
        
        # Parse the LLM response
        parsed = _parse_llm_response(analysis_result)
        
        # Determine if model passes
        rejection_threshold = get_rejection_threshold()
        passed = parsed["confidence_hardcoded"] < rejection_threshold
        
        result = {
            "passed": passed,
            "confidence_hardcoded": parsed["confidence_hardcoded"],
            "red_flags": parsed["red_flags"],
            "evidence": parsed["evidence"],
            "model_used": DETECTION_MODEL_ID,
            "analysis_cost_usd": cost
        }
        
        # Log the result
        if passed:
            logger.info(
                f"   ✅ Hardcoding check PASSED: {parsed['confidence_hardcoded']}% confidence "
                f"(threshold: {rejection_threshold}%)"
            )
        else:
            logger.warning(
                f"   ❌ Hardcoding check FAILED: {parsed['confidence_hardcoded']}% confidence "
                f"(threshold: {rejection_threshold}%)"
            )
            logger.warning(f"   Red flags: {parsed['red_flags']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Hardcoding detection error: {e}")
        # Allow the model to run on error (don't block legitimate submissions)
        return {
            "passed": True,
            "confidence_hardcoded": 0,
            "red_flags": [],
            "evidence": f"Detection error: {str(e)}",
            "model_used": None,
            "analysis_cost_usd": 0.0
        }


# =============================================================================
# Helper Functions
# =============================================================================

async def _extract_code_from_tarball(model_code: bytes) -> Tuple[Optional[str], int, int]:
    """
    Extract files from tarball for analysis.
    
    ALL files count toward size limit (py, md, txt, json, etc.)
    Only .py files are sent to LLM for hardcoding analysis.
    
    Returns:
        Tuple of (combined_py_content, py_file_count, total_size_all_files)
        Returns (None, 0, 0) if extraction fails
    """
    try:
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            f.write(model_code)
            temp_path = f.name
        
        # Extract ALL files and track total size
        py_files = {}
        total_size = 0
        
        with tarfile.open(temp_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode("utf-8", errors="ignore")
                        total_size += len(content.encode('utf-8'))
                        
                        # Only keep .py files for LLM analysis
                        if member.name.endswith(".py"):
                            py_files[member.name] = content
        
        # Clean up
        Path(temp_path).unlink()
        
        if not py_files:
            return None, 0, total_size
        
        # Combine .py files for LLM analysis
        combined = []
        for filepath, content in sorted(py_files.items()):
            combined.append(f"# ===== FILE: {filepath} =====")
            combined.append(content)
            combined.append("")
        
        return "\n".join(combined), len(py_files), total_size
        
    except Exception as e:
        logger.error(f"Error extracting code from tarball: {e}")
        return None, 0, 0


def _build_analysis_prompt(code_content: str, icp_samples: List[Dict[str, Any]]) -> str:
    """
    Build the prompt for hardcoding detection.
    
    The core question: Does this model's output genuinely depend on the ICP input?
    Would it work on a brand new ICP it has never seen before?
    """
    
    # Format the actual ICP samples that will be tested
    icp_json_samples = ""
    for i, icp in enumerate(icp_samples[:5], 1):  # Show up to 5 ICPs
        # Extract key fields
        icp_display = {
            "industry": icp.get("industry", ""),
            "sub_industry": icp.get("sub_industry", ""),
            "target_roles": icp.get("target_roles", icp.get("target_role_titles", [])),
            "geography": icp.get("geography", icp.get("target_geography", "")),
            "company_size": icp.get("company_size", icp.get("company_size_range", "")),
        }
        icp_json_samples += f"\n**ICP {i}:**\n```json\n{json.dumps(icp_display, indent=2)}\n```\n"
    
    prompt = f"""## Competition Overview

This is a B2B lead qualification agent competition. Models receive an ICP (Ideal Customer Profile) and must find a matching sales lead by querying the leads database (table name provided via config).

**What a legitimate model does:**
1. Parse the ICP to understand what kind of lead is needed
2. Query the database with filters derived from the ICP (industry, geography, etc.)
3. Use APIs to enrich or validate the lead
4. Return a lead that matches the ICP criteria

**The test:** ICPs are generated fresh daily. The model must generalize to ANY ICP.

## ICPs That Will Be Tested

These are ACTUAL ICPs this model will be evaluated against:
{icp_json_samples}

## Code Submission

```python
{code_content}
```

## Your Analysis Task

**Phase 1: Check for blatant hardcoding**

- Does the code return static/hardcoded emails, names, or companies?
- Are there lookup tables mapping specific industries to specific leads?
- Does it ignore the ICP and return the same output every time?
- Does it generate fake data without querying the database?
- Does it use random name generators (e.g., `random.choice(FIRST_NAMES)`) to fabricate person data even if company data is from DB?

**Also flag if you see obvious malicious patterns** (auto-fail, 100% confidence):
- Code that downloads and executes remote payloads, or opens network connections to unknown servers
- Attempts to read files outside the working directory (especially paths containing `.bittensor`, `.ssh`, `.aws`, or `/etc/`)

**Phase 2: Trace ICP flow through the code**

- Is `icp["industry"]` used in database queries?
- Is `icp["target_roles"]` used to filter or score candidates?
- Is `icp["geography"]` used meaningfully?
- Would different ICPs produce different queries?

**Phase 3: Check against the actual test ICPs above**

- Does the code have hardcoded answers for any of the industries shown above?
- Would it produce meaningfully different results for each ICP?

## Confidence Scale

- **0-30%**: Legitimate - ICP flows through to database queries and API calls
- **31-69%**: Suspicious - Partial ICP usage mixed with hardcoded elements  
- **70-100%**: Hardcoded - Output doesn't depend on ICP input

## Response

```json
{{
    "confidence_hardcoded": <0-100>,
    "red_flags": ["specific issue found", ...],
    "evidence": "How ICP flows (or doesn't flow) through the code",
    "verdict": "LEGITIMATE|SUSPICIOUS|HARDCODED"
}}
```"""
    
    return prompt


async def _call_reasoning_llm(
    prompt: str,
    api_key: str,
    timeout: float = 120.0
) -> Tuple[str, float]:
    """
    Call OpenRouter API with o3-mini for hardcoding detection.
    
    Returns:
        Tuple of (response_text, cost_usd)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://leadpoet.io",
        "X-Title": "LeadPoet Hardcoding Detection"
    }
    
    payload = {
        "model": DETECTION_MODEL_ID,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,  # Zero temp for deterministic results
        "max_tokens": 4000
    }
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Extract response text
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract cost from response
        cost = 0.0
        usage = data.get("usage", {})
        if "cost" in usage:
            cost = usage["cost"]
        elif "total_cost" in usage:
            cost = usage["total_cost"]
        else:
            # Estimate cost from tokens using o3-mini pricing
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            cost = (
                (input_tokens / 1_000_000 * DETECTION_MODEL_COST_INPUT) +
                (output_tokens / 1_000_000 * DETECTION_MODEL_COST_OUTPUT)
            )
        
        return content, cost


def _parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse the LLM's JSON response."""
    try:
        # Try to extract JSON from the response
        # The LLM might include markdown code blocks
        json_str = response
        
        if "```json" in response:
            # Extract JSON from code block
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                json_str = response[start:end].strip()
        elif "```" in response:
            # Try generic code block
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                json_str = response[start:end].strip()
        
        # Parse JSON
        parsed = json.loads(json_str)
        
        # Validate and normalize
        confidence = int(parsed.get("confidence_hardcoded", 50))
        confidence = max(0, min(100, confidence))  # Clamp to 0-100
        
        return {
            "confidence_hardcoded": confidence,
            "red_flags": parsed.get("red_flags", []),
            "evidence": parsed.get("evidence", "No explanation provided"),
            "verdict": parsed.get("verdict", "UNKNOWN")
        }
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        logger.warning(f"Raw response: {response[:500]}...")
        
        # Try to extract confidence from response text
        confidence = 50  # Default to uncertain
        if "hardcoded" in response.lower():
            confidence = 75
        elif "legitimate" in response.lower():
            confidence = 25
        
        return {
            "confidence_hardcoded": confidence,
            "red_flags": ["Could not parse LLM response"],
            "evidence": response[:1000],
            "verdict": "UNKNOWN"
        }


# =============================================================================
# Utility Functions for Coordinator/Worker Integration
# =============================================================================

def should_skip_hardcoding_check(model_data: Dict[str, Any]) -> bool:
    """
    Determine if hardcoding check should be skipped for this model.
    
    Skip for:
    - Champion rebenchmarks (already passed check when first submitted)
    - Models that have already been evaluated before
    """
    # Check if this is a rebenchmark
    is_rebenchmark = model_data.get("is_rebenchmark", False)
    if is_rebenchmark:
        return True
    
    # Check if model has been evaluated before (re-evaluation)
    evaluated_at = model_data.get("evaluated_at")
    if evaluated_at:
        return True
    
    return False
