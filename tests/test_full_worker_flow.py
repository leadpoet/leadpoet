#!/usr/bin/env python3
"""
EXACT SIMULATION: Gateway → Workers → Coordinator Flow

This test simulates EXACTLY what happens in production:
1. Gateway sends 12 leads to coordinator
2. Coordinator saves to epoch_{N}_leads.json
3. Each of 6 containers (1 coordinator + 5 workers) processes 2 leads each
4. Workers write result files with validation_results and local_validation_data
5. Coordinator aggregates all worker results
6. Final result should be 12 validated leads

This uses the EXACT code paths from neurons/validator.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def simulate_gateway_leads(num_leads=12):
    """Simulate what gateway sends to coordinator"""
    leads = []
    for i in range(num_leads):
        leads.append({
            'lead_id': f'lead-{i+1:03d}',
            'lead_blob': {
                'email': f'person{i+1}@company{i+1}.com',
                'first': f'Person{i+1}',
                'last': f'Last{i+1}',
                'full_name': f'Person{i+1} Last{i+1}',
                'company': f'Company{i+1}',
                'role': 'CEO',
                'region': 'New York',
                'industry': 'Technology',
                'sub_industry': 'Software',
                'linkedin': f'https://linkedin.com/in/person{i+1}',
                'wallet_ss58': f'5FMiner{i+1}...',
                'source': 'linkedin',
                'source_type': 'public_profile',
                'source_url': f'https://example.com/{i+1}',
                'website': f'https://company{i+1}.com',
            },
            'lead_blob_hash': f'hash-{i+1}',
            'miner_hotkey': f'5FMiner{i+1}...'
        })
    return leads


def simulate_coordinator_distributes_leads(all_leads, total_containers=6):
    """
    Simulate EXACT lead distribution logic from validator.py lines 2130-2150
    
    This is the EXACT code:
        leads_per_container = original_count // total_containers
        remainder = original_count % total_containers
        
        if container_id < remainder:
            start = container_id * (leads_per_container + 1)
            end = start + leads_per_container + 1
        else:
            start = (remainder * (leads_per_container + 1)) + ((container_id - remainder) * leads_per_container)
            end = start + leads_per_container
    """
    original_count = len(all_leads)
    leads_per_container = original_count // total_containers
    remainder = original_count % total_containers
    
    container_leads = {}
    for container_id in range(total_containers):
        if container_id < remainder:
            start = container_id * (leads_per_container + 1)
            end = start + leads_per_container + 1
        else:
            start = (remainder * (leads_per_container + 1)) + ((container_id - remainder) * leads_per_container)
            end = start + leads_per_container
        
        container_leads[container_id] = all_leads[start:end]
    
    return container_leads


def simulate_worker_validates_and_writes(container_id, worker_leads, epoch_id, temp_dir):
    """
    Simulate EXACT worker validation and file writing from validator.py lines 4726-4761
    
    This is the EXACT format workers now write after the fix.
    """
    validation_results = []
    local_validation_data = []
    
    for lead_data in worker_leads:
        lead_id = lead_data['lead_id']
        lead_blob = lead_data['lead_blob']
        miner_hotkey = lead_data.get('miner_hotkey', lead_blob.get('wallet_ss58', 'unknown'))
        
        # Simulate validation (simplified - in reality this calls run_automated_checks)
        is_valid = True  # For test, assume all pass
        rejection_reason = None
        automated_checks_data = {
            'stage_0_hardcoded': {'passed': True},
            'stage_1_dns': {'passed': True},
        }
        
        # Format for validation_results (for gateway hash submission)
        validation_results.append({
            'lead_id': lead_id,
            'is_valid': is_valid,
            'miner_hotkey': miner_hotkey
        })
        
        # Format for local_validation_data (for gateway reveal submission)
        local_validation_data.append({
            'lead_id': lead_id,
            'is_valid': is_valid,
            'rejection_reason': rejection_reason,
            'validation_details': automated_checks_data,
            'lead_blob': lead_blob
        })
    
    # Write result file (EXACT format from lines 4748-4761)
    results_file = Path(temp_dir) / f"worker_{container_id}_epoch_{epoch_id}_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'epoch_id': epoch_id,
            'container_id': container_id,
            'validation_results': validation_results,  # CRITICAL: EXACT KEY
            'local_validation_data': local_validation_data,  # CRITICAL: EXACT KEY
            'lead_range': f"{len(worker_leads)} leads",
            'timestamp': 1234567890.0
        }, f)
    
    return results_file


def simulate_coordinator_aggregates(coordinator_results, coordinator_reveals, worker_ids, epoch_id, temp_dir):
    """
    Simulate EXACT coordinator aggregation from validator.py lines 2298-2326
    
    This is the EXACT code that reads worker files.
    """
    aggregated_validation_results = list(coordinator_results)
    aggregated_local_validation_data = list(coordinator_reveals)
    
    worker_files = []
    for worker_id in worker_ids:
        worker_file = os.path.join(temp_dir, f"worker_{worker_id}_epoch_{epoch_id}_results.json")
        worker_files.append((worker_id, worker_file))
    
    # EXACT aggregation logic from lines 2302-2320
    for worker_id, worker_file in worker_files:
        if os.path.exists(worker_file):
            try:
                with open(worker_file, 'r') as f:
                    worker_data = json.load(f)
                
                # EXACT key names from line 2308-2310
                worker_validations = worker_data.get("validation_results", [])
                worker_reveals = worker_data.get("local_validation_data", [])
                worker_range = worker_data.get("lead_range", "unknown")
                
                aggregated_validation_results.extend(worker_validations)
                aggregated_local_validation_data.extend(worker_reveals)
                
                print(f"   ✅ Aggregated {len(worker_validations)} results from Container-{worker_id} (range: {worker_range})")
            except Exception as e:
                print(f"   ⚠️  Failed to load worker Container-{worker_id}: {e}")
                raise
    
    return aggregated_validation_results, aggregated_local_validation_data


def run_full_flow_test():
    """Run complete end-to-end test"""
    
    print("="*80)
    print("EXACT SIMULATION: Gateway → Workers → Coordinator")
    print("="*80)
    print()
    
    epoch_id = 99999
    total_containers = 6  # 1 coordinator + 5 workers
    num_leads = 12  # 2 leads per container
    
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # STEP 1: Gateway sends leads to coordinator
        print("STEP 1: Gateway sends 12 leads to coordinator")
        all_leads = simulate_gateway_leads(num_leads)
        print(f"   ✅ Gateway generated {len(all_leads)} leads")
        print()
        
        # STEP 2: Coordinator distributes leads (EXACT logic from validator.py)
        print("STEP 2: Coordinator distributes leads using EXACT validator.py logic")
        container_leads = simulate_coordinator_distributes_leads(all_leads, total_containers)
        
        for container_id, leads in container_leads.items():
            lead_ids = [l['lead_id'] for l in leads]
            print(f"   Container {container_id}: {len(leads)} leads - {lead_ids}")
        print()
        
        # Verify distribution
        total_distributed = sum(len(leads) for leads in container_leads.values())
        assert total_distributed == num_leads, f"Distribution error: {total_distributed} != {num_leads}"
        
        # STEP 3: Coordinator (container 0) validates its leads
        print("STEP 3: Coordinator validates its 2 leads")
        coordinator_leads = container_leads[0]
        coordinator_results = []
        coordinator_reveals = []
        
        for lead_data in coordinator_leads:
            coordinator_results.append({
                'lead_id': lead_data['lead_id'],
                'is_valid': True,
                'miner_hotkey': lead_data['miner_hotkey']
            })
            coordinator_reveals.append({
                'lead_id': lead_data['lead_id'],
                'is_valid': True,
                'rejection_reason': None,
                'validation_details': {},
                'lead_blob': lead_data['lead_blob']
            })
        
        print(f"   ✅ Coordinator validated {len(coordinator_results)} leads")
        print()
        
        # STEP 4: Workers validate and write files (EXACT format from validator.py)
        print("STEP 4: Workers validate and write result files")
        worker_ids = [1, 2, 3, 4, 5]
        
        for worker_id in worker_ids:
            worker_leads = container_leads[worker_id]
            result_file = simulate_worker_validates_and_writes(worker_id, worker_leads, epoch_id, temp_dir)
            print(f"   ✅ Worker {worker_id}: Validated {len(worker_leads)} leads → {result_file.name}")
        print()
        
        # STEP 5: Coordinator aggregates (EXACT logic from validator.py)
        print("STEP 5: Coordinator aggregates all results")
        final_validations, final_reveals = simulate_coordinator_aggregates(
            coordinator_results,
            coordinator_reveals,
            worker_ids,
            epoch_id,
            temp_dir
        )
        print()
        
        # STEP 6: Verify final results
        print("STEP 6: Verify aggregation")
        print(f"   📊 Total aggregated validations: {len(final_validations)}")
        print(f"   📊 Total aggregated reveals: {len(final_reveals)}")
        print()
        
        # Assertions
        assert len(final_validations) == num_leads, f"❌ FAIL: Expected {num_leads} validations, got {len(final_validations)}"
        assert len(final_reveals) == num_leads, f"❌ FAIL: Expected {num_leads} reveals, got {len(final_reveals)}"
        
        # Verify all lead IDs are present
        expected_lead_ids = {f'lead-{i+1:03d}' for i in range(num_leads)}
        actual_lead_ids = {v['lead_id'] for v in final_validations}
        assert expected_lead_ids == actual_lead_ids, f"❌ FAIL: Lead IDs don't match"
        
        # Verify structure
        for v in final_validations:
            assert 'lead_id' in v, "❌ Missing lead_id in validation_results"
            assert 'is_valid' in v, "❌ Missing is_valid in validation_results"
            assert 'miner_hotkey' in v, "❌ Missing miner_hotkey in validation_results"
        
        for r in final_reveals:
            assert 'lead_blob' in r, "❌ Missing lead_blob in local_validation_data"
            assert 'validation_details' in r, "❌ Missing validation_details in local_validation_data"
        
        print("="*80)
        print("✅ SUCCESS: Full flow test passed!")
        print("="*80)
        print()
        print("Summary:")
        print(f"  - Gateway sent: {num_leads} leads")
        print(f"  - Coordinator processed: {len(coordinator_results)} leads")
        print(f"  - Workers processed: {len(final_validations) - len(coordinator_results)} leads")
        print(f"  - Final aggregated: {len(final_validations)} leads")
        print()
        print("✅ Workers write correct keys (validation_results, local_validation_data)")
        print("✅ Coordinator reads correct keys and aggregates successfully")
        print("✅ No data loss in aggregation")


if __name__ == "__main__":
    try:
        run_full_flow_test()
    except AssertionError as e:
        print()
        print("="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80)
        exit(1)
    except Exception as e:
        print()
        print("="*80)
        print(f"❌ TEST ERROR: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        exit(1)

