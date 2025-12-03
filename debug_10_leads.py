#!/usr/bin/env python3
"""
Debug Script - Trace EXACTLY what happens for each lead
========================================================
Shows:
1. All DDG search results (raw)
2. Which result is selected for fuzzy match
3. Fuzzy match conclusion
4. Exact LLM input prompt
5. LLM raw response
"""
import asyncio
import sys
import os
import csv
from datetime import datetime

sys.path.insert(0, '.')

# Load env
env_path = '.env'
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ[key] = val

from validator_models.lead_verification import verify_stage5_unified

# Load leads from CSV
CSV_PATH = "/Users/pranav/Downloads/Supabase Snippet APPROVED LEADS CSV (1).csv"

def load_leads_from_csv():
    leads = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            leads.append({
                'full_name': row.get('full_name', ''),
                'company': row.get('business', ''),
                'linkedin': row.get('linkedin', ''),
                'role': row.get('role', ''),
                'region': row.get('region', ''),
                'industry': row.get('industry', ''),
                'sub_industry': row.get('sub_industry', ''),
                'website': row.get('website', ''),
            })
    return leads

# The EXACT 10 leads from previous tests
EXACT_TEST_NAMES = [
    "Mathilde Patmon",
    "Brandon Rutledge", 
    "Bingrui Yang",
    "Sam Dresser",
    "Spenser Skates",
    "Jason Dann",
    "Dustin Deus",
    "Jack Scari",
    "Abhinav Joshi",
    "Ray Mauritsson",
]

print("=" * 120)
print("DEBUG TRACE - FULL TRANSPARENCY FOR 10 LEADS")
print("=" * 120)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Load and find exact leads
all_leads = load_leads_from_csv()
test_leads = []
for name in EXACT_TEST_NAMES:
    for lead in all_leads:
        if lead.get('full_name', '').strip() == name:
            test_leads.append(lead)
            break

print(f"Testing {len(test_leads)} leads")
print()

async def run_debug():
    for i, lead in enumerate(test_leads, 1):
        full_name = lead.get("full_name", "")
        company = lead.get("company", "")
        linkedin = lead.get("linkedin", "")
        miner_role = lead.get("role", "")
        miner_region = lead.get("region", "")
        miner_industry = lead.get("industry", "")
        
        print()
        print("#" * 120)
        print(f"# LEAD {i}/10: {full_name} @ {company}")
        print("#" * 120)
        print(f"LinkedIn: {linkedin}")
        print(f"Miner Role: {miner_role}")
        print(f"Miner Region: {miner_region}")
        print(f"Miner Industry: {miner_industry}")
        print()
        
        # Call verify_stage5_unified with verbose=True to get debug data
        result = await verify_stage5_unified(lead, None, verbose=True)
        
        # Extract debug data
        debug = result.get("_debug", {})
        
        # =====================================================================
        # 1. ALL DDG SEARCH RESULTS (RAW)
        # =====================================================================
        print("=" * 100)
        print("1. DDG SEARCH RESULTS (RAW)")
        print("=" * 100)
        
        print("\n--- ROLE RESULTS ---")
        role_results = debug.get("ddg_role_results", [])
        if role_results:
            for j, r in enumerate(role_results, 1):
                print(f"  [{j}] Title: {r.get('title', '')[:80]}")
                print(f"      Snippet: {r.get('snippet', r.get('body', ''))[:100]}")
                print(f"      Query: {r.get('query', 'N/A')}")
                print()
        else:
            print("  (no results)")
        
        print("\n--- REGION RESULTS ---")
        region_results = debug.get("ddg_region_results", [])
        if region_results:
            for j, r in enumerate(region_results[:3], 1):
                print(f"  [{j}] Title: {r.get('title', '')[:80]}")
                print(f"      Snippet: {r.get('snippet', r.get('body', ''))[:100]}")
        else:
            print("  (no results)")
        
        print("\n--- INDUSTRY RESULTS ---")
        industry_results = debug.get("ddg_industry_results", [])
        if industry_results:
            for j, r in enumerate(industry_results[:3], 1):
                print(f"  [{j}] Title: {r.get('title', '')[:80]}")
                print(f"      Snippet: {r.get('snippet', r.get('body', ''))[:100]}")
        else:
            print("  (no results)")
        
        # =====================================================================
        # 2. FUZZY MATCH SELECTION
        # =====================================================================
        print()
        print("=" * 100)
        print("2. FUZZY MATCH DETAILS")
        print("=" * 100)
        
        fuzzy_result = debug.get("fuzzy_result", {})
        print(f"\n  ROLE:")
        print(f"    Verified by fuzzy: {fuzzy_result.get('role_verified', 'N/A')}")
        print(f"    Extracted: {fuzzy_result.get('role_extracted', 'N/A')}")
        print(f"    Confidence: {fuzzy_result.get('role_confidence', 'N/A')}")
        print(f"    Reason: {fuzzy_result.get('role_reason', 'N/A')}")
        
        print(f"\n  REGION:")
        print(f"    Verified by fuzzy: {fuzzy_result.get('region_verified', 'N/A')}")
        print(f"    Extracted: {fuzzy_result.get('region_extracted', 'N/A')}")
        print(f"    Confidence: {fuzzy_result.get('region_confidence', 'N/A')}")
        print(f"    Reason: {fuzzy_result.get('region_reason', 'N/A')}")
        
        print(f"\n  INDUSTRY:")
        print(f"    Verified by fuzzy: {fuzzy_result.get('industry_verified', 'N/A')}")
        print(f"    (Always sent to LLM)")
        
        # =====================================================================
        # 3. WHAT FIELDS NEED LLM
        # =====================================================================
        print()
        print("=" * 100)
        print("3. FIELDS NEEDING LLM VERIFICATION")
        print("=" * 100)
        needs_llm = debug.get("needs_llm", [])
        print(f"  Fields sent to LLM: {needs_llm}")
        
        # =====================================================================
        # 4. EXACT LLM INPUT PROMPT
        # =====================================================================
        print()
        print("=" * 100)
        print("4. EXACT LLM INPUT PROMPT")
        print("=" * 100)
        llm_prompt = debug.get("llm_prompt")
        if llm_prompt:
            print(llm_prompt)
        else:
            print("  (No LLM prompt - all fields verified by fuzzy match)")
        
        # =====================================================================
        # 5. LLM RAW RESPONSE
        # =====================================================================
        print()
        print("=" * 100)
        print("5. LLM RAW RESPONSE")
        print("=" * 100)
        llm_raw = debug.get("llm_response_raw")
        if llm_raw:
            print(llm_raw)
        else:
            print("  (No LLM response - all fields verified by fuzzy match)")
        
        # =====================================================================
        # FINAL RESULT
        # =====================================================================
        print()
        print("=" * 100)
        print("FINAL RESULT")
        print("=" * 100)
        print(f"  Overall: {'✅ PASS' if result.get('verified') else '❌ FAIL'}")
        print(f"  Role: {'✅' if result.get('role_match') else '❌'} | Extracted: {result.get('extracted_role', 'N/A')}")
        print(f"  Region: {'✅' if result.get('region_match') else '❌'} | Extracted: {result.get('extracted_region', 'N/A')}")
        print(f"  Industry: {'✅' if result.get('industry_match') else '❌'} | Extracted: {result.get('extracted_industry', 'N/A')}")
        
        print()
        print("-" * 120)

asyncio.run(run_debug())

print()
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

