"""
Load Test Harness for Leadpoet API
Tests API performance at 250 QPS / 1000 burst capacity.
Target: Sustains load within SLA (400ms P95, 550ms P99)
"""

import asyncio
import aiohttp
import time
import statistics
import json
import logging
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import random

logger = logging.getLogger(__name__)


@dataclass
class LoadTestResult:
    """Results from a load test run."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration: float
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_rate: float
    sla_violations: int
    cost_per_lead: float


class LoadTestHarness:
    """Load test harness for Leadpoet API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        
        # Test queries for variety
        self.test_queries = [
            {
                "query": "Find CRM software for technology companies with 100-500 employees in the US",
                "desired_count": 25,
                "user_id": "load-test-user-1"
            },
            {
                "query": "Looking for marketing automation tools for SaaS companies in Europe",
                "desired_count": 15,
                "user_id": "load-test-user-2"
            },
            {
                "query": "ERP systems for manufacturing companies with 500+ employees",
                "desired_count": 20,
                "user_id": "load-test-user-3"
            },
            {
                "query": "HR software for healthcare companies, but not for startups",
                "desired_count": 10,
                "user_id": "load-test-user-4"
            },
            {
                "query": "Analytics platforms for finance companies similar to Tableau",
                "desired_count": 30,
                "user_id": "load-test-user-5"
            }
        ]
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def make_request(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single API request."""
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.base_url}/query",
                json=query_data,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=10.0)
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    response_data = await response.json()
                    return {
                        "success": True,
                        "response_time": response_time,
                        "status_code": response.status,
                        "leads_count": len(response_data),
                        "cost": self._estimate_cost(response_data)
                    }
                else:
                    return {
                        "success": False,
                        "response_time": response_time,
                        "status_code": response.status,
                        "error": f"HTTP {response.status}"
                    }
                    
        except asyncio.TimeoutError:
            return {
                "success": False,
                "response_time": time.time() - start_time,
                "status_code": 408,
                "error": "Timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "response_time": time.time() - start_time,
                "status_code": 500,
                "error": str(e)
            }
    
from app.core.config import settings

    def _estimate_cost(self, leads: List[Dict[str, Any]]) -> float:
        """Estimate cost based on leads returned."""
        # Use actual cost settings for accurate estimation
        base_cost_per_lead = settings.BASE_COST_PER_LEAD
        llm_cost_per_lead = settings.LLM_COST_PER_CALL / len(leads) if leads else 0

        total_cost = 0.0
        for lead in leads:
            if lead.get("llm_score") is not None:
                total_cost += base_cost_per_lead + llm_cost_per_lead
            else:
                total_cost += base_cost_per_lead

        return total_cost    
    async def run_sustained_load_test(self, target_qps: int = 250, duration_seconds: int = 60) -> LoadTestResult:
        """Run sustained load test at target QPS."""
        logger.info(f"Starting sustained load test: {target_qps} QPS for {duration_seconds} seconds")
        
        start_time = time.time()
        request_interval = 1.0 / target_qps
        results = []
        
        # Calculate total requests needed
        total_requests = target_qps * duration_seconds
        
        for i in range(total_requests):
            # Calculate the exact time when this request should be made
            next_request_time = start_time + (i * request_interval)
            
            # Select random query
            query_data = random.choice(self.test_queries).copy()
            
            # Make request
            result = await self.make_request(query_data)
            results.append(result)
            
            # Calculate sleep time to maintain consistent intervals
            if i < total_requests - 1:  # Don't wait after last request
                current_time = time.time()
                sleep_duration = next_request_time + request_interval - current_time
                
                # Only sleep if we're behind schedule (positive sleep duration)
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)
                else:
                    # Log if we're falling behind schedule
                    logger.warning(f"Request {i+1} is {abs(sleep_duration):.3f}s behind schedule")
        
        end_time = time.time()
        return self._analyze_results(results, end_time - start_time)
    
    async def run_burst_load_test(self, burst_size: int = 1000, concurrency: int = 50) -> LoadTestResult:
        """Run burst load test with specified concurrency."""
        logger.info(f"Starting burst load test: {burst_size} requests with {concurrency} concurrent workers")
        
        start_time = time.time()
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        async def make_request_with_semaphore():
            async with semaphore:
                query_data = random.choice(self.test_queries).copy()
                return await self.make_request(query_data)
        
        # Create all tasks
        tasks = [make_request_with_semaphore() for _ in range(burst_size)]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "response_time": 0.0,
                    "status_code": 500,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        end_time = time.time()
        return self._analyze_results(processed_results, end_time - start_time)
    
    def _analyze_results(self, results: List[Dict[str, Any]], total_duration: float) -> LoadTestResult:
        """Analyze test results and calculate metrics."""
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        response_times = [r["response_time"] for r in successful_requests]
        
        # Calculate percentiles using numpy for accurate interpolation
        if response_times:
            # Convert to numpy array for accurate percentile calculation
            response_times_array = np.array(response_times)
            
            # Calculate percentiles with proper interpolation
            p50 = np.percentile(response_times_array, 50)
            p95 = np.percentile(response_times_array, 95)
            p99 = np.percentile(response_times_array, 99)
            avg_response_time = np.mean(response_times_array)
        else:
            p50 = p95 = p99 = avg_response_time = 0.0
        
        # Calculate SLA violations (400ms P95, 550ms P99)
        sla_violations = sum(1 for rt in response_times if rt > 0.4)  # 400ms threshold
        
        # Calculate cost metrics
        total_cost = sum(r.get("cost", 0.0) for r in successful_requests)
        total_leads = sum(r.get("leads_count", 0) for r in successful_requests)
        cost_per_lead = total_cost / total_leads if total_leads > 0 else 0.0
        
        return LoadTestResult(
            total_requests=len(results),
            successful_requests=len(successful_requests),
            failed_requests=len(failed_requests),
            total_duration=total_duration,
            avg_response_time=avg_response_time,
            p50_response_time=p50,
            p95_response_time=p95,
            p99_response_time=p99,
            requests_per_second=len(results) / total_duration,
            error_rate=len(failed_requests) / len(results) if results else 0.0,
            sla_violations=sla_violations,
            cost_per_lead=cost_per_lead
        )
    
    def print_results(self, result: LoadTestResult, test_name: str):
        """Print formatted test results."""
        print(f"\n{'='*60}")
        print(f"LOAD TEST RESULTS: {test_name}")
        print(f"{'='*60}")
        print(f"Total Requests: {result.total_requests}")
        print(f"Successful: {result.successful_requests}")
        print(f"Failed: {result.failed_requests}")
        print(f"Error Rate: {result.error_rate:.2%}")
        print(f"Duration: {result.total_duration:.2f}s")
        print(f"Requests/sec: {result.requests_per_second:.2f}")
        print(f"\nResponse Times:")
        print(f"  Average: {result.avg_response_time*1000:.2f}ms")
        print(f"  P50: {result.p50_response_time*1000:.2f}ms")
        print(f"  P95: {result.p95_response_time*1000:.2f}ms")
        print(f"  P99: {result.p99_response_time*1000:.2f}ms")
        print(f"\nSLA Compliance:")
        print(f"  P95 SLA (400ms): {'✅ PASS' if result.p95_response_time <= 0.4 else '❌ FAIL'}")
        print(f"  P99 SLA (550ms): {'✅ PASS' if result.p99_response_time <= 0.55 else '❌ FAIL'}")
        print(f"  SLA Violations: {result.sla_violations}")
        print(f"\nCost Metrics:")
        print(f"  Cost per Lead: ${result.cost_per_lead:.6f}")
        print(f"  Budget Compliance (<$0.002): {'✅ PASS' if result.cost_per_lead <= 0.002 else '❌ FAIL'}")


async def main():
    """Main function to run load tests."""
    async with LoadTestHarness() as harness:
        # Test 1: Sustained load at 250 QPS
        print("Running sustained load test...")
        sustained_result = await harness.run_sustained_load_test(target_qps=250, duration_seconds=30)
        harness.print_results(sustained_result, "Sustained Load (250 QPS)")
        
        # Test 2: Burst load with 1000 requests
        print("\nRunning burst load test...")
        burst_result = await harness.run_burst_load_test(burst_size=1000, concurrency=50)
        harness.print_results(burst_result, "Burst Load (1000 requests)")
        
        # Overall assessment
        print(f"\n{'='*60}")
        print("OVERALL ASSESSMENT")
        print(f"{'='*60}")
        
        sustained_pass = (sustained_result.p95_response_time <= 0.4 and 
                         sustained_result.p99_response_time <= 0.55 and
                         sustained_result.cost_per_lead <= 0.002)
        
        burst_pass = (burst_result.p95_response_time <= 0.4 and 
                     burst_result.p99_response_time <= 0.55 and
                     burst_result.cost_per_lead <= 0.002)
        
        if sustained_pass and burst_pass:
            print("✅ ALL TESTS PASSED - API meets SLA requirements")
        else:
            print("❌ SOME TESTS FAILED - API needs optimization")
            if not sustained_pass:
                print("  - Sustained load test failed SLA requirements")
            if not burst_pass:
                print("  - Burst load test failed SLA requirements")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main()) 