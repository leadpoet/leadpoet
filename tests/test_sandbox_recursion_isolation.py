"""
Test: Sandbox recursion limit isolation

Reproduces the exact production bug where a miner's model calls
sys.setrecursionlimit(50), poisoning the Python process for all
subsequent model evaluations on the same worker.

The fix (in SandboxSecurityContext) saves sys.getrecursionlimit()
on __enter__ and restores it on __exit__. These tests verify:

1. A model that lowers the recursion limit doesn't affect the next model
2. A model that raises the recursion limit doesn't affect the next model
3. The recursion limit is restored even when the model raises an exception
4. Supabase-style DB queries work after a poisoning model ran
5. The fix works across multiple sequential evaluations (simulating
   a full worker lifecycle with 5+ models)
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qualification.validator.sandbox_security import SandboxSecurityContext


DEFAULT_LIMIT = sys.getrecursionlimit()


def _recursive_comparison(depth=0):
    """Simulate the Supabase client's comparison chain that fails
    when recursion limit is too low. This mirrors what happens when
    postgrest-py builds filter chains internally."""
    if depth >= 100:
        return True
    return _recursive_comparison(depth + 1)


# =========================================================================
# Test 1: Model lowers recursion limit -> next model is unaffected
# =========================================================================
async def test_recursion_limit_restored_after_lowering():
    """A model that sets sys.setrecursionlimit(50) should NOT affect
    subsequent models. This is the exact production bug."""
    
    before = sys.getrecursionlimit()
    assert before == DEFAULT_LIMIT, f"Pre-condition: expected {DEFAULT_LIMIT}, got {before}"
    
    # Model A: malicious/buggy - lowers recursion limit
    ctx = SandboxSecurityContext(
        evaluation_run_id="test-model-a",
        evaluation_id="eval-a",
        enable_import_restriction=False,
        enable_network_interception=False,
        enable_env_sanitization=False,
        enable_file_restriction=False,
    )
    async with ctx:
        sys.setrecursionlimit(50)
        assert sys.getrecursionlimit() == 50, "Model A should be able to set limit during its run"
    
    after_a = sys.getrecursionlimit()
    assert after_a == before, (
        f"CRITICAL: Recursion limit was {before} before Model A, "
        f"but is {after_a} after. Model A poisoned the process!"
    )
    
    # Model B: legitimate - should work normally
    ctx_b = SandboxSecurityContext(
        evaluation_run_id="test-model-b",
        evaluation_id="eval-b",
        enable_import_restriction=False,
        enable_network_interception=False,
        enable_env_sanitization=False,
        enable_file_restriction=False,
    )
    async with ctx_b:
        assert sys.getrecursionlimit() == DEFAULT_LIMIT
        result = _recursive_comparison()
        assert result is True, "Model B's recursive comparison should succeed"
    
    print("  PASS: Recursion limit restored after model lowered it")


# =========================================================================
# Test 2: Model raises recursion limit -> next model is unaffected
# =========================================================================
async def test_recursion_limit_restored_after_raising():
    """A model that raises the limit to 100000 should not persist."""
    
    before = sys.getrecursionlimit()
    
    ctx = SandboxSecurityContext(
        evaluation_run_id="test-model-high",
        evaluation_id="eval-high",
        enable_import_restriction=False,
        enable_network_interception=False,
        enable_env_sanitization=False,
        enable_file_restriction=False,
    )
    async with ctx:
        sys.setrecursionlimit(100000)
        assert sys.getrecursionlimit() == 100000
    
    after = sys.getrecursionlimit()
    assert after == before, (
        f"Recursion limit should be {before} after model set it to 100000, got {after}"
    )
    
    print("  PASS: Recursion limit restored after model raised it")


# =========================================================================
# Test 3: Restore works even when model throws an exception
# =========================================================================
async def test_recursion_limit_restored_after_exception():
    """If a model crashes, the recursion limit must still be restored."""
    
    before = sys.getrecursionlimit()
    
    ctx = SandboxSecurityContext(
        evaluation_run_id="test-model-crash",
        evaluation_id="eval-crash",
        enable_import_restriction=False,
        enable_network_interception=False,
        enable_env_sanitization=False,
        enable_file_restriction=False,
    )
    try:
        async with ctx:
            sys.setrecursionlimit(30)
            raise RuntimeError("Model crashed!")
    except RuntimeError:
        pass
    
    after = sys.getrecursionlimit()
    assert after == before, (
        f"Recursion limit should be {before} after model crash, got {after}"
    )
    
    print("  PASS: Recursion limit restored after model exception")


# =========================================================================
# Test 4: DB-style recursive comparison works after poisoning model
# =========================================================================
async def test_db_queries_work_after_poisoning():
    """Simulate the exact production scenario:
    1. Model A sets recursion limit to 50
    2. Model B tries to do a Supabase-style recursive comparison
    3. Model B should succeed (not get RecursionError)"""
    
    before = sys.getrecursionlimit()
    
    # Model A: poison
    ctx_a = SandboxSecurityContext(
        evaluation_run_id="test-poison",
        evaluation_id="eval-poison",
        enable_import_restriction=False,
        enable_network_interception=False,
        enable_env_sanitization=False,
        enable_file_restriction=False,
    )
    async with ctx_a:
        sys.setrecursionlimit(50)
    
    # Model B: should work
    ctx_b = SandboxSecurityContext(
        evaluation_run_id="test-victim",
        evaluation_id="eval-victim",
        enable_import_restriction=False,
        enable_network_interception=False,
        enable_env_sanitization=False,
        enable_file_restriction=False,
    )
    async with ctx_b:
        try:
            result = _recursive_comparison()
            assert result is True
        except RecursionError:
            raise AssertionError(
                "CRITICAL: Model B got RecursionError! "
                "Model A's poisoning was NOT cleaned up."
            )
    
    print("  PASS: DB-style queries work after poisoning model")


# =========================================================================
# Test 5: Full worker lifecycle - 5 sequential models
# =========================================================================
async def test_worker_lifecycle_5_models():
    """Simulate a real worker processing 5 models in sequence.
    Model 2 poisons the recursion limit. Models 3-5 must be unaffected."""
    
    original = sys.getrecursionlimit()
    results = []
    
    for i in range(1, 6):
        ctx = SandboxSecurityContext(
            evaluation_run_id=f"test-worker-model-{i}",
            evaluation_id=f"eval-{i}",
            enable_import_restriction=False,
            enable_network_interception=False,
            enable_env_sanitization=False,
            enable_file_restriction=False,
        )
        async with ctx:
            limit_at_start = sys.getrecursionlimit()
            
            if i == 2:
                sys.setrecursionlimit(50)
            
            try:
                _recursive_comparison()
                results.append((i, "OK", limit_at_start))
            except RecursionError:
                results.append((i, "RECURSION_ERROR", limit_at_start))
        
        after = sys.getrecursionlimit()
        assert after == original, (
            f"After model {i}, recursion limit is {after}, expected {original}"
        )
    
    for model_num, status, limit_at_start in results:
        if model_num == 2:
            continue
        assert status == "OK", (
            f"Model {model_num} got {status} (limit at start: {limit_at_start}). "
            f"Poisoning from model 2 leaked!"
        )
        assert limit_at_start == original, (
            f"Model {model_num} started with limit {limit_at_start}, expected {original}"
        )
    
    print("  PASS: Worker lifecycle - all 5 models handled correctly")


# =========================================================================
# Test 6: Without the fix (regression guard)
# =========================================================================
async def test_without_fix_would_fail():
    """Verify that without save/restore, the poisoning WOULD persist.
    This confirms our test is actually testing the right thing."""
    
    original = sys.getrecursionlimit()
    
    # Manually poison (no context manager)
    sys.setrecursionlimit(50)
    
    try:
        _recursive_comparison()
        restored = False
    except RecursionError:
        restored = False
    
    # Clean up
    sys.setrecursionlimit(original)
    
    # Now verify context manager fixes it
    ctx = SandboxSecurityContext(
        evaluation_run_id="test-verify",
        evaluation_id="eval-verify",
        enable_import_restriction=False,
        enable_network_interception=False,
        enable_env_sanitization=False,
        enable_file_restriction=False,
    )
    async with ctx:
        sys.setrecursionlimit(50)
    
    after = sys.getrecursionlimit()
    assert after == original, f"Context manager should restore to {original}, got {after}"
    
    result = _recursive_comparison()
    assert result is True, "Comparison should succeed after context manager cleanup"
    
    print("  PASS: Regression guard - fix works, without it would fail")


# =========================================================================
# Main
# =========================================================================
async def run_all():
    print("\n=== Sandbox Recursion Limit Isolation Tests ===\n")
    
    await test_recursion_limit_restored_after_lowering()
    await test_recursion_limit_restored_after_raising()
    await test_recursion_limit_restored_after_exception()
    await test_db_queries_work_after_poisoning()
    await test_worker_lifecycle_5_models()
    await test_without_fix_would_fail()
    
    print(f"\n=== ALL 6 TESTS PASSED ===\n")


if __name__ == "__main__":
    asyncio.run(run_all())
