"""
Test Runner for Leadpoet Intent Model
Executes all tests and generates coverage reports.
Target: ≥80% coverage across pipeline
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print("❌ FAILED")
            if result.stderr:
                print("Error:")
                print(result.stderr)
            if result.stdout:
                print("Output:")
                print(result.stdout)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def install_test_dependencies():
    """Install test dependencies."""
    dependencies = [
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "coverage",
        "aiohttp",
        "redis",
        "prometheus-client"
    ]
    
    for dep in dependencies:
        success = run_command(f"pip install {dep}", f"Installing {dep}")
        if not success:
            print(f"Failed to install {dep}")
            return False
    
    return True

def run_unit_tests():
    """Run all unit tests."""
    test_files = [
        "test_prompt_parser.py",
        "test_scoring_formula.py",
        "test_cost_validation.py",
        "test_coverage.py"
    ]
    
    all_passed = True
    
    for test_file in test_files:
        test_path = os.path.join(os.path.dirname(__file__), test_file)
        if os.path.exists(test_path):
            success = run_command(
                f"python -m pytest {test_file} -v",
                f"Unit tests: {test_file}"
            )
            if not success:
                all_passed = False
        else:
            print(f"⚠️  Test file not found: {test_file}")
    
    return all_passed

def run_coverage_analysis():
    """Run coverage analysis."""
    # Run coverage with pytest
    success = run_command(
        "python -m pytest --cov=app --cov-report=html --cov-report=term-missing --cov-report=json",
        "Coverage analysis with pytest"
    )
    
    if success:
        # Parse coverage report
        coverage_file = "coverage.json"
        alt_coverage_file = "htmlcov/coverage.json"
        
        # Check both possible locations
        if not os.path.exists(coverage_file) and os.path.exists(alt_coverage_file):
            coverage_file = alt_coverage_file
            
        if os.path.exists(coverage_file):
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)
            
            total_coverage = coverage_data.get('totals', {}).get('percent_covered', 0)
            print(f"\n📊 Total Coverage: {total_coverage:.1f}%")
            
            if total_coverage >= 80.0:
                print("✅ Coverage target achieved (≥80%)")
                return True
            else:
                print("❌ Coverage target not met (<80%)")
                return False
        else:
            print("⚠️  Coverage report file not found")
            return False
def run_load_tests():
    """Run load tests."""
    load_test_file = "load_test_harness.py"
    load_test_path = os.path.join(os.path.dirname(__file__), load_test_file)
    
    if os.path.exists(load_test_path):
        success = run_command(
            f"python {load_test_file}",
            "Load testing (250 QPS / 1000 burst)"
        )
        return success
    else:
        print(f"⚠️  Load test file not found: {load_test_file}")
        return False

def run_cost_validation():
    """Run cost validation tests."""
    cost_test_file = "test_cost_validation.py"
    cost_test_path = os.path.join(os.path.dirname(__file__), cost_test_file)
    
    if os.path.exists(cost_test_path):
        success = run_command(
            f"python -m pytest {cost_test_file}::test_cost_validation_report -v",
            "Cost validation (<$0.002 per lead)"
        )
        return success
    else:
        print(f"⚠️  Cost validation file not found: {cost_test_file}")
        return False

def generate_test_report():
    """Generate comprehensive test report."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {
            "unit_tests": False,
            "coverage_analysis": False,
            "load_tests": False,
            "cost_validation": False
        },
        "coverage_percentage": 0.0,
        "overall_status": "FAILED"
    }
    
    print("\n" + "="*80)
    print("LEADPOET INTENT MODEL - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Install dependencies
    print("\n📦 Installing test dependencies...")
    if not install_test_dependencies():
        print("❌ Failed to install dependencies")
        return report
    
    # Run unit tests
    print("\n🧪 Running unit tests...")
    report["tests"]["unit_tests"] = run_unit_tests()
    
    # Run coverage analysis
    print("\n📊 Running coverage analysis...")
    coverage_success = run_coverage_analysis()
    report["tests"]["coverage_analysis"] = coverage_success
    
    # Run load tests
    print("\n⚡ Running load tests...")
    report["tests"]["load_tests"] = run_load_tests()
    
    # Run cost validation
    print("\n💰 Running cost validation...")
    report["tests"]["cost_validation"] = run_cost_validation()
    
    # Determine overall status
    all_passed = all(report["tests"].values())
    report["overall_status"] = "PASSED" if all_passed else "FAILED"
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in report["tests"].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Status: {report['overall_status']}")
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Unit tests completed successfully")
        print("✅ Coverage target achieved (≥80%)")
        print("✅ Load tests passed (250 QPS / 1000 burst)")
        print("✅ Cost validation passed (<$0.002 per lead)")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        failed_tests = [name for name, passed in report["tests"].items() if not passed]
        for test in failed_tests:
            print(f"❌ {test.replace('_', ' ').title()} failed")
    
    # Save report
    report_file = "test_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Test report saved to: {report_file}")
    
    return report

def main():
    """Main function."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "unit":
            run_unit_tests()
        elif command == "coverage":
            run_coverage_analysis()
        elif command == "load":
            run_load_tests()
        elif command == "cost":
            run_cost_validation()
        elif command == "all":
            generate_test_report()
        else:
            print("Usage: python run_tests.py [unit|coverage|load|cost|all]")
            print("  unit     - Run unit tests only")
            print("  coverage - Run coverage analysis only")
            print("  load     - Run load tests only")
            print("  cost     - Run cost validation only")
            print("  all      - Run all tests (default)")
    else:
        generate_test_report()

if __name__ == "__main__":
    main() 