"""
Test script for query quality reputation persistence functionality.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.database import SessionLocal
from app.services.query_quality_reputation_service import QueryQualityReputationService
from app.core.metrics import MetricsCollector
from app.models.query_quality_reputation import QueryQualityReputation, QuerySubmitterThrottle
from app.models.query import Query
from datetime import datetime, timedelta
import time


def test_query_quality_reputation_persistence():
    """Test query quality reputation persistence functionality."""
    print("🧪 Testing query quality reputation persistence...")
    
    # Create database session
    db = SessionLocal()
    metrics_collector = MetricsCollector()
    
    try:
        # Create reputation service
        reputation_service = QueryQualityReputationService(db, metrics_collector)
        
        # Test user ID
        test_user_id = "test_user_001"
        
        print(f"📊 Testing reputation calculation for user: {test_user_id}")
        
        # Simulate query results for reputation calculation
        query_result = {"status": "passed"}  # Simulate a passed query
        query_quality_score = 0.85  # Simulate a good quality score
        
        # Update reputation (this should create a new record)
        quality_score, pass_rate, flag_rate = reputation_service.update_reputation(
            test_user_id, query_result, query_quality_score
        )
        
        print(f"   Initial quality score: {quality_score:.3f}")
        print(f"   Pass rate: {pass_rate:.3f}")
        print(f"   Flag rate: {flag_rate:.3f}")
        
        # Verify the record was saved to database
        db_reputation = db.query(QueryQualityReputation).filter(
            QueryQualityReputation.user_id == test_user_id
        ).first()
        
        if db_reputation:
            print(f"✅ Reputation saved to database: {db_reputation.query_quality_score:.3f}")
            print(f"   Total queries: {db_reputation.total_queries}")
            print(f"   Passed queries: {db_reputation.passed_queries}")
            print(f"   Flagged queries: {db_reputation.flagged_queries}")
        else:
            print("❌ Reputation not found in database")
            return False
        
        # Test throttle functionality
        print(f"🚦 Testing throttle functionality...")
        
        # Simulate poor performance to trigger throttling
        for i in range(5):
            # Simulate flagged queries
            query_result = {"status": "flagged"}
            query_quality_score = 0.3  # Low quality score
            reputation_service.update_reputation(test_user_id, query_result, query_quality_score)
        
        # Check throttle status
        throttle_info = reputation_service.get_throttle_info(test_user_id)
        if throttle_info and throttle_info['is_throttled']:
            print(f"✅ User throttled successfully")
            print(f"   Throttle reason: {throttle_info['throttle_reason']}")
            print(f"   Throttle end time: {throttle_info['throttle_end_time']}")
        else:
            print("❌ User not throttled when expected")
            return False
        
        # Test clearing expired throttle
        print(f"🧹 Testing throttle clearing...")
        
        # Manually expire the throttle by setting end time to past
        db_throttle = db.query(QuerySubmitterThrottle).filter(
            QuerySubmitterThrottle.user_id == test_user_id
        ).first()
        if db_throttle:
            db_throttle.throttle_end_time = datetime.utcnow() - timedelta(hours=1)
            db.commit()
        
        # Clear expired throttles
        cleared_count = reputation_service.clear_expired_throttles()
        print(f"   Cleared {cleared_count} expired throttles")
        
        # Verify throttle is cleared
        throttle_info_after = reputation_service.get_throttle_info(test_user_id)
        if not throttle_info_after or not throttle_info_after['is_throttled']:
            print("✅ Expired throttle cleared successfully")
        else:
            print("❌ Throttle not cleared properly")
            return False
        
        # Test EMA calculation with previous reputation
        print(f"📈 Testing EMA calculation...")
        
        # Update reputation again (should use previous reputation for EMA)
        query_result = {"status": "passed"}
        query_quality_score = 0.9  # High quality score
        quality_score2, pass_rate2, flag_rate2 = reputation_service.update_reputation(
            test_user_id, query_result, query_quality_score
        )
        
        print(f"   Updated quality score: {quality_score2:.3f}")
        print(f"   Updated pass rate: {pass_rate2:.3f}")
        print(f"   Updated flag rate: {flag_rate2:.3f}")
        
        # Verify EMA was used (quality score should be smoothed)
        # Verify EMA was used (quality score should be smoothed)
        # With alpha=0.1, new score should be: old_score * 0.9 + new_score * 0.1
        expected_score = quality_score * 0.9 + query_quality_score * 0.1
        if abs(quality_score2 - expected_score) < 0.001:
            print(f"✅ EMA calculation working correctly (expected: {expected_score:.3f}, actual: {quality_score2:.3f})")
        else:
            print(f"❌ EMA calculation incorrect (expected: {expected_score:.3f}, actual: {quality_score2:.3f})")
            return False        # Test reputation summary
        print(f"📋 Testing reputation summary...")
        summary = reputation_service.get_reputation_summary()
        print(f"   Total users: {summary['total_users']}")
        print(f"   Throttled users: {summary['throttled_users']}")
        print(f"   Average quality score: {summary['average_quality_score']:.3f}")
        print(f"   Average flag rate: {summary['average_flag_rate']:.3f}")
        
        print("🎉 All persistence tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test data
        try:
            db.query(QueryQualityReputation).filter(
                QueryQualityReputation.user_id == test_user_id
            ).delete()
            db.query(QuerySubmitterThrottle).filter(
                QuerySubmitterThrottle.user_id == test_user_id
            ).delete()
            db.commit()
            print("🧹 Test data cleaned up")
        except Exception as e:
            print(f"⚠️  Error cleaning up test data: {e}")
        
        db.close()


if __name__ == "__main__":
    success = test_query_quality_reputation_persistence()
    sys.exit(0 if success else 1) 