#!/usr/bin/env python3
"""
Test script for automated lead validation using leads-100.csv
Independent test of the automated_checks.py module
"""

import asyncio
import csv
import sys
import os
from datetime import datetime

# Add the current directory to Python path so we can import from validator_models
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validator_models.automated_checks import run_automated_checks


def load_leads_from_csv(filepath: str) -> list:
    """Load leads from CSV file and return as list of dictionaries"""
    leads = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            # Read CSV with DictReader to automatically map column names
            csv_reader = csv.DictReader(file)
            
            for row in csv_reader:
                # Convert CSV row to lead dictionary matching the expected format
                lead = {
                    "Index": row.get("Index", ""),
                    "Account Id": row.get("Account Id", ""),
                    "Lead Owner": row.get("Lead Owner", ""),
                    "First Name": row.get("First Name", ""),
                    "Last Name": row.get("Last Name", ""),
                    "Company": row.get("Company", ""),
                    "Phone 1": row.get("Phone 1", ""),
                    "Phone 2": row.get("Phone 2", ""),
                    "Email 1": row.get("Email 1", ""),
                    "Email 2": row.get("Email 2", ""),
                    "Website": row.get("Website", ""),
                    "Source": row.get("Source", ""),
                    "Deal Stage": row.get("Deal Stage", ""),
                    "Notes": row.get("Notes", "")
                }
                leads.append(lead)
                
        return leads
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find CSV file: {filepath}")
        return []
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return []


async def run_individual_checks(leads: list):
    """Run individual automated checks on each lead"""
    
    print(f"🔍 Running automated checks on {len(leads)} leads...")
    print("=" * 80)
    
    results = []
    
    for i, lead in enumerate(leads):
        email = lead.get("Email 1", "")
        company = lead.get("Company", "")
        
        try:
            # Run the automated checks for this individual lead
            # The run_automated_checks function already has all the detailed logging built in
            passed, reason = await run_automated_checks(lead)
            
            status = "Valid" if passed else "Invalid"
            results.append({
                "lead_index": i + 1,  # 1-based indexing to match CSV
                "email": email,
                "company": company,
                "status": status,
                "reason": reason
            })
            
        except Exception as e:
            results.append({
                "lead_index": i + 1,
                "email": email,
                "company": company,
                "status": "Invalid",
                "reason": f"Processing error: {str(e)}"
            })
    
    return results


def print_results_summary(results: list):
    """Print results summary similar to extended-test-logs.txt"""
    
    print("✅ Automated checks completed successfully!")
    print()
    print("📊 Results:")
    print("-" * 60)
    
    valid_count = 0
    invalid_count = 0
    
    for result in results:
        status_icon = "✅" if result["status"] == "Valid" else "❌"
        print(f"{status_icon} Lead {result['lead_index']}: {result['status']} - {result['reason']}")
        print(f"   Email: {result['email']}")
        
        if result["status"] == "Valid":
            valid_count += 1
        else:
            invalid_count += 1
    
    print()
    print("🎯 Summary:")
    print(f"   Total Leads: {len(results)}")
    print(f"   Valid: {valid_count}")
    print(f"   Invalid: {invalid_count}")
    
    success_rate = (valid_count / len(results)) * 100 if results else 0
    print(f"   Success Rate: {success_rate:.1f}%")
    print()


async def main():
    """Main function to run the test workflow"""
    
    print("🚀 Automated Checks - Natural Workflow Test")
    print("=" * 80)
    
    # Load leads from CSV
    csv_file = "leads-100.csv"
    leads = load_leads_from_csv(csv_file)
    
    if not leads:
        print("❌ No leads loaded. Exiting.")
        return
    
    print(f"✅ Loaded {len(leads)} leads from {csv_file}")
    print("✅ Successfully imported automated checks")
    print()
    
    # Run individual checks to get detailed output like extended-test-logs.txt
    results = await run_individual_checks(leads)
    
    # Print summary
    print_results_summary(results)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
