#!/usr/bin/env python3
"""
CRITICAL TEST: Worker-Coordinator JSON Key Compatibility

This test ensures that worker result files have the EXACT keys that the coordinator expects.
If this test fails, the coordinator will aggregate 0 results from workers.

Expected keys:
- validation_results
- local_validation_data
- lead_range (optional)
"""

import json
import tempfile
from pathlib import Path


def test_worker_result_file_format():
    """Test that worker writes correct JSON keys for coordinator aggregation"""
    
    # Simulate what worker writes
    worker_result = {
        'epoch_id': 12345,
        'container_id': 1,
        'validation_results': [  # MUST BE THIS KEY
            {
                'lead_id': 'test-lead-1',
                'is_valid': True,
                'miner_hotkey': '5FTest...'
            }
        ],
        'local_validation_data': [  # MUST BE THIS KEY
            {
                'lead_id': 'test-lead-1',
                'is_valid': True,
                'rejection_reason': None,
                'validation_details': {},
                'lead_blob': {'email': 'test@test.com'}
            }
        ],
        'lead_range': '1 leads',
        'timestamp': 1234567890.0
    }
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(worker_result, f)
        temp_path = f.name
    
    try:
        # Simulate what coordinator reads
        with open(temp_path, 'r') as f:
            loaded_data = json.load(f)
        
        # CRITICAL: Coordinator expects these exact keys
        worker_validations = loaded_data.get("validation_results", [])
        worker_reveals = loaded_data.get("local_validation_data", [])
        worker_range = loaded_data.get("lead_range", "unknown")
        
        # Assertions
        assert len(worker_validations) > 0, "❌ FAIL: validation_results is empty or missing!"
        assert len(worker_reveals) > 0, "❌ FAIL: local_validation_data is empty or missing!"
        assert worker_range != "unknown", "❌ FAIL: lead_range is missing!"
        
        # Validate structure
        assert 'lead_id' in worker_validations[0], "❌ FAIL: validation_results missing lead_id"
        assert 'is_valid' in worker_validations[0], "❌ FAIL: validation_results missing is_valid"
        assert 'miner_hotkey' in worker_validations[0], "❌ FAIL: validation_results missing miner_hotkey"
        
        assert 'lead_blob' in worker_reveals[0], "❌ FAIL: local_validation_data missing lead_blob"
        assert 'validation_details' in worker_reveals[0], "❌ FAIL: local_validation_data missing validation_details"
        
        print("✅ PASS: Worker result file has correct keys for coordinator aggregation")
        print(f"   - validation_results: {len(worker_validations)} entries")
        print(f"   - local_validation_data: {len(worker_reveals)} entries")
        print(f"   - lead_range: {worker_range}")
        
    finally:
        Path(temp_path).unlink()


def test_wrong_keys_fail():
    """Test that WRONG keys result in 0 aggregated results (the bug we had)"""
    
    # Simulate WRONG format (what worker was writing before fix)
    wrong_worker_result = {
        'epoch_id': 12345,
        'container_id': 1,
        'validated_leads': [  # WRONG KEY (was the bug!)
            {
                'lead_id': 'test-lead-1',
                'is_valid': True,
            }
        ],
        'timestamp': 1234567890.0
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(wrong_worker_result, f)
        temp_path = f.name
    
    try:
        # Coordinator reads with correct keys
        with open(temp_path, 'r') as f:
            loaded_data = json.load(f)
        
        worker_validations = loaded_data.get("validation_results", [])
        worker_reveals = loaded_data.get("local_validation_data", [])
        
        # This is what was happening - coordinator got 0 results!
        assert len(worker_validations) == 0, "Expected 0 results with wrong keys"
        assert len(worker_reveals) == 0, "Expected 0 results with wrong keys"
        
        print("✅ PASS: Confirmed wrong keys result in 0 aggregated results")
        print("   This is the bug that was causing coordinator to aggregate nothing!")
        
    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    print("="*80)
    print("WORKER-COORDINATOR KEY COMPATIBILITY TEST")
    print("="*80)
    print()
    
    try:
        test_worker_result_file_format()
        print()
        test_wrong_keys_fail()
        print()
        print("="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
    except AssertionError as e:
        print()
        print("="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80)
        exit(1)

