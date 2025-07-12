"""
Query Quality Reputation Service for Leadpoet Intent Model v1.1
Manages query quality reputation tracking and throttling for query submitters.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.models.query_quality_reputation import QueryQualityReputation, QuerySubmitterThrottle
from app.core.metrics import MetricsCollector
from app.core.config import settings
from loguru import logger


class QueryQualityReputationService:
    """
    Service for managing query quality reputation and throttling.
    
    Tracks query quality metrics for users (miners/validators) and implements
    throttling based on performance thresholds.
    """
    
    def __init__(self, db: Session, metrics: MetricsCollector):
        self.db = db
        self.metrics = metrics
        
        # EMA parameters for reputation calculation
        self.alpha = 0.1  # Smoothing factor (0.1 = 10% weight to new values)
        self.min_queries_for_reputation = 10  # Minimum queries before calculating reputation
        
        # Throttling thresholds
        self.flag_rate_threshold = 0.3  # 30% flag rate triggers throttling
        self.quality_score_threshold = 0.5  # Quality score below 0.5 triggers throttling
        self.throttle_duration_hours = 24  # Default throttle duration
        
        # Alert thresholds
        self.alert_flag_rate_threshold = 0.5  # 50% flag rate triggers alert
        self.alert_quality_score_threshold = 0.3  # Quality score below 0.3 triggers alert
    
    def get_reputation(self, user_id: str) -> Optional[QueryQualityReputation]:
        """Get query quality reputation for a user."""
        return self.db.query(QueryQualityReputation).filter(
            QueryQualityReputation.user_id == user_id
        ).first()
    
    def get_throttle_status(self, user_id: str) -> Optional[QuerySubmitterThrottle]:
        """Get throttling status for a user."""
        return self.db.query(QuerySubmitterThrottle).filter(
            QuerySubmitterThrottle.user_id == user_id
        ).first()
    
    def update_reputation(
        self, 
        user_id: str, 
        query_result: Dict,
        query_quality_score: float
    ) -> Tuple[float, float, float]:
        """
        Update query quality reputation for a user.
        
        Args:
            user_id: User identifier
            query_result: Query result with pass/flag status
            query_quality_score: Quality score for the query (0.0-1.0)
            
        Returns:
            Tuple of (query_quality_score, pass_rate, flag_rate)
        """
        # Validate input
        if not 0.0 <= query_quality_score <= 1.0:
            raise ValueError(f"query_quality_score must be between 0.0 and 1.0, got {query_quality_score}")

        # Get or create reputation record
        reputation = self.get_reputation(user_id)
        if not reputation:
            reputation = QueryQualityReputation(
                user_id=user_id,
                query_quality_score=1.0,
                pass_rate=1.0,
                flag_rate=0.0,
                total_queries=0,
                passed_queries=0,
                flagged_queries=0
            )
            self.db.add(reputation)
        
        # Update query counts
        reputation.total_queries += 1
        
        # Determine if query passed or was flagged
        is_passed = query_result.get('status') == 'passed'
        if is_passed:
            reputation.passed_queries += 1
        else:
            reputation.flagged_queries += 1
        
        # Calculate new rates
        new_pass_rate = reputation.passed_queries / reputation.total_queries
        new_flag_rate = reputation.flagged_queries / reputation.total_queries
        
        # Update EMA values
        if reputation.total_queries >= self.min_queries_for_reputation:
            reputation.query_quality_score = (
                self.alpha * query_quality_score + 
                (1 - self.alpha) * reputation.query_quality_score
            )
            reputation.pass_rate = (
                self.alpha * new_pass_rate + 
                (1 - self.alpha) * reputation.pass_rate
            )
            reputation.flag_rate = (
                self.alpha * new_flag_rate + 
                (1 - self.alpha) * reputation.flag_rate
            )
        else:
            # Use simple averages for first few queries
            reputation.query_quality_score = (
                (reputation.query_quality_score * (reputation.total_queries - 1) + query_quality_score) / 
                reputation.total_queries
            )
            reputation.pass_rate = new_pass_rate
            reputation.flag_rate = new_flag_rate
        
        reputation.last_calculation_time = datetime.utcnow()
        
        # Update metrics
        self.metrics.query_quality_score_gauge.labels(user_id=user_id).set(reputation.query_quality_score)
        self.metrics.query_pass_rate_gauge.labels(user_id=user_id).set(reputation.pass_rate)
        self.metrics.query_flag_rate_gauge.labels(user_id=user_id).set(reputation.flag_rate)
        self.metrics.query_total_queries_gauge.labels(user_id=user_id).set(reputation.total_queries)
        
        # Check for throttling
        self._check_and_update_throttling(user_id, reputation)
        
        # Check for alerts
        self._check_alerts(user_id, reputation)
        
        self.db.commit()
        
        logger.info(
            f"Updated query quality reputation for {user_id}: "
            f"score={reputation.query_quality_score:.3f}, "
            f"pass_rate={reputation.pass_rate:.3f}, "
            f"flag_rate={reputation.flag_rate:.3f}"
        )
        
        return reputation.query_quality_score, reputation.pass_rate, reputation.flag_rate
    
    def _check_and_update_throttling(self, user_id: str, reputation: QueryQualityReputation):
        """Check if user should be throttled and update status."""
        throttle = self.get_throttle_status(user_id)
        if not throttle:
            throttle = QuerySubmitterThrottle(user_id=user_id)
            self.db.add(throttle)
        
        # Check if user should be throttled
        should_throttle = (
            reputation.flag_rate > self.flag_rate_threshold or
            reputation.query_quality_score < self.quality_score_threshold
        )
        
        # Check if currently throttled
        currently_throttled = throttle.is_currently_throttled()
        
        if should_throttle and not currently_throttled:
            # Start throttling
            throttle.is_throttled = True
            throttle.throttle_start_time = datetime.utcnow()
            throttle.throttle_end_time = datetime.utcnow() + timedelta(hours=self.throttle_duration_hours)
            throttle.throttle_count += 1
            
            # Record reason
            if reputation.flag_rate > self.flag_rate_threshold:
                throttle.throttle_reason = "high_flag_rate"
            else:
                throttle.throttle_reason = "low_quality"
            
            throttle.flag_rate_at_throttle = reputation.flag_rate
            throttle.quality_score_at_throttle = reputation.query_quality_score
            
            # Update total throttle duration
            throttle.total_throttle_duration_hours += self.throttle_duration_hours
            
            # Update metrics
            self.metrics.query_submitter_throttled_gauge.labels(user_id=user_id).set(1)
            
            logger.warning(
                f"Throttled user {user_id} for {self.throttle_duration_hours}h: "
                f"flag_rate={reputation.flag_rate:.3f}, "
                f"quality_score={reputation.query_quality_score:.3f}"
            )
        
        elif not should_throttle and currently_throttled:
            # Remove throttling
            throttle.is_throttled = False
            throttle.throttle_start_time = None
            throttle.throttle_end_time = None
            throttle.throttle_reason = None
            
            # Update metrics
            self.metrics.query_submitter_throttled_gauge.labels(user_id=user_id).set(0)
            
            logger.info(f"Removed throttling for user {user_id}")
    
    def _check_alerts(self, user_id: str, reputation: QueryQualityReputation):
        """Check for alert conditions and increment alert counter."""
        alert_triggered = (
            reputation.flag_rate > self.alert_flag_rate_threshold or
            reputation.query_quality_score < self.alert_quality_score_threshold
        )
        
        if alert_triggered:
            self.metrics.query_quality_alert_counter.labels(user_id=user_id).inc()
            
            logger.error(
                f"Query quality alert for user {user_id}: "
                f"flag_rate={reputation.flag_rate:.3f}, "
                f"quality_score={reputation.query_quality_score:.3f}"
            )
    
    def is_user_throttled(self, user_id: str) -> bool:
        """Check if a user is currently throttled."""
        throttle = self.get_throttle_status(user_id)
        if not throttle:
            return False
        
        return throttle.is_currently_throttled()
    
    def get_throttle_info(self, user_id: str) -> Optional[Dict]:
        """Get detailed throttle information for a user."""
        throttle = self.get_throttle_status(user_id)
        if not throttle:
            return None
        
        return {
            'is_throttled': throttle.is_currently_throttled(),
            'throttle_start_time': throttle.throttle_start_time.isoformat() if throttle.throttle_start_time else None,
            'throttle_end_time': throttle.throttle_end_time.isoformat() if throttle.throttle_end_time else None,
            'remaining_time_seconds': throttle.get_remaining_throttle_time(),
            'throttle_reason': throttle.throttle_reason,
            'throttle_count': throttle.throttle_count,
            'total_throttle_duration_hours': throttle.total_throttle_duration_hours
        }
    
    def clear_expired_throttles(self) -> int:
        """Clear expired throttles and return count of cleared throttles."""
        now = datetime.utcnow()
        
        expired_throttles = self.db.query(QuerySubmitterThrottle).filter(
            and_(
                QuerySubmitterThrottle.is_throttled == True,
                QuerySubmitterThrottle.throttle_end_time < now
            )
        ).all()
        
        cleared_count = 0
        for throttle in expired_throttles:
            throttle.is_throttled = False
            throttle.throttle_start_time = None
            throttle.throttle_end_time = None
            throttle.throttle_reason = None
            
            # Update metrics
            self.metrics.query_submitter_throttled_gauge.labels(user_id=throttle.user_id).set(0)
            
            cleared_count += 1
        
        if cleared_count > 0:
            self.db.commit()
        # Calculate average metrics
        from sqlalchemy import func
        
        avg_quality = self.db.query(
            func.avg(QueryQualityReputation.query_quality_score)
        ).scalar() or 0.0
        
        avg_flag_rate = self.db.query(
            func.avg(QueryQualityReputation.flag_rate)
        ).scalar() or 0.0
        
        return cleared_count
    
    def get_all_throttles(self, limit: int = 100) -> List[Dict]:
        """Get all throttle statuses with optional limit."""
        throttles = self.db.query(QuerySubmitterThrottle).limit(limit).all()
        return [throttle.to_dict() for throttle in throttles]
    
    def get_reputation_summary(self) -> Dict:
        """Get summary statistics for all reputations."""
        total_users = self.db.query(QueryQualityReputation).count()
        throttled_users = self.db.query(QuerySubmitterThrottle).filter(
            QuerySubmitterThrottle.is_throttled == True
        ).count()
        
        # Calculate average metrics
        avg_quality = self.db.query(
            QueryQualityReputation.query_quality_score
        ).scalar() or 0.0
        
        avg_flag_rate = self.db.query(
            QueryQualityReputation.flag_rate
        ).scalar() or 0.0
        
        return {
            'total_users': total_users,
            'throttled_users': throttled_users,
            'average_quality_score': avg_quality,
            'average_flag_rate': avg_flag_rate
        } 