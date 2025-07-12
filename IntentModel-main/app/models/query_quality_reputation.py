"""
Query Quality Reputation Model for Leadpoet Intent Model v1.1
Stores query quality reputation data for EMA calculations and persistence across restarts.
"""

from sqlalchemy import Column, String, Float, DateTime, Integer, Boolean, Text
from sqlalchemy.sql import func
from app.core.database import Base
from datetime import datetime, timezone

# app/models/query_quality_reputation.py

class QueryQualityReputation(Base):
    """Database model for storing query quality reputation data."""
    
    __tablename__ = "query_quality_reputation"
    
    # Primary key - user identifier (could be miner or validator)
    user_id = Column(String(255), primary_key=True, index=True)
    
    # Reputation data
    query_quality_score = Column(Float, nullable=False, default=1.0)
    pass_rate = Column(Float, nullable=False, default=1.0)
    flag_rate = Column(Float, nullable=False, default=0.0)
    
    # Query statistics
    total_queries = Column(Integer, nullable=False, default=0)
    passed_queries = Column(Integer, nullable=False, default=0)
    flagged_queries = Column(Integer, nullable=False, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Metadata
    last_calculation_time = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    calculation_window_days = Column(Integer, nullable=False, default=30)  # Days to look back for calculations
    
    def __repr__(self):
        return f"<QueryQualityReputation(user_id='{self.user_id}', query_quality_score={self.query_quality_score:.3f})>"
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'user_id': self.user_id,
            'query_quality_score': self.query_quality_score,
            'pass_rate': self.pass_rate,
            'flag_rate': self.flag_rate,
            'total_queries': self.total_queries,
            'passed_queries': self.passed_queries,
            'flagged_queries': self.flagged_queries,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_calculation_time': self.last_calculation_time.isoformat() if self.last_calculation_time else None,
            'calculation_window_days': self.calculation_window_days
        }


class QuerySubmitterThrottle(Base):
    """Database model for storing query submitter throttling data."""
    
    __tablename__ = "query_submitter_throttles"
    
    # Primary key - user identifier (could be miner or validator)
    user_id = Column(String(255), primary_key=True, index=True)
    
    # Throttling data
    is_throttled = Column(Boolean, nullable=False, default=False)
    throttle_start_time = Column(DateTime(timezone=True), nullable=True)
    throttle_end_time = Column(DateTime(timezone=True), nullable=True)
    
    # Throttling reason and metadata
    throttle_reason = Column(String(255), nullable=True)  # e.g., "high_flag_rate", "low_quality"
    flag_rate_at_throttle = Column(Float, nullable=True)
    quality_score_at_throttle = Column(Float, nullable=True)
    
    # Throttling history
    throttle_count = Column(Integer, nullable=False, default=0)
    total_throttle_duration_hours = Column(Float, nullable=False, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<QuerySubmitterThrottle(user_id='{self.user_id}', is_throttled={self.is_throttled})>"
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'user_id': self.user_id,
            'is_throttled': self.is_throttled,
            'throttle_start_time': self.throttle_start_time.isoformat() if self.throttle_start_time else None,
            'throttle_end_time': self.throttle_end_time.isoformat() if self.throttle_end_time else None,
            'throttle_reason': self.throttle_reason,
            'flag_rate_at_throttle': self.flag_rate_at_throttle,
            'quality_score_at_throttle': self.quality_score_at_throttle,
            'throttle_count': self.throttle_count,
            'total_throttle_duration_hours': self.total_throttle_duration_hours,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    def is_currently_throttled(self) -> bool:
        """Check if user is currently throttled based on time."""
        if not self.is_throttled or not self.throttle_end_time:
            return False
        
        return datetime.now(timezone.utc) < self.throttle_end_time.replace(tzinfo=timezone.utc)
    
    def get_remaining_throttle_time(self) -> float:
        """Get remaining throttle time in seconds."""
        if not self.is_throttled or not self.throttle_end_time:
            return 0.0
        
        remaining = (
            self.throttle_end_time.replace(tzinfo=timezone.utc)
            - datetime.now(timezone.utc)
        ).total_seconds()
        return max(0.0, remaining) 