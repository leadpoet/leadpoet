import uuid
from typing import Union
from sqlalchemy.orm import Session
from loguru import logger

from app.core.config import settings
from app.models.query import Query

class QueryPerformanceService:
    """
    Service for calculating and updating a query's performance score.
    """
    def __init__(self, db: Session):
        self.db = db
        self.ctr_weight = settings.QUERY_PERF_CTR_WEIGHT
        self.conversion_weight = settings.QUERY_PERF_CONVERSION_WEIGHT
        self.score_threshold = settings.QUERY_PERF_SCORE_THRESHOLD

    def calculate_performance_score(self, ctr: float, conversion_rate: float) -> float:
        """
        Calculates the performance score based on Click-Through Rate (CTR) and conversion rate.
        """
        score = (self.ctr_weight * ctr) + (self.conversion_weight * conversion_rate)
        return max(0.0, min(1.0, score))

    def update_query_performance(self, query_id: str, ctr: float, conversion_rate: float):
        """
        Updates a query's performance score and flags it if the score is below the configured threshold.
        """
        query = self.db.query(Query).filter(Query.id == query_id).first()
        if not query:
            logger.warning(f"Query with id {query_id} not found for performance update.")
            return

        new_score = self.calculate_performance_score(ctr, conversion_rate)
        
        query.mqf_score = new_score # The DB column name remains mqf_score for now to avoid a new migration

        if new_score < self.score_threshold:
            if not query.is_flagged:
                query.is_flagged = True
                logger.info(f"Query {query_id} has been flagged due to a low performance score: {new_score:.2f}")
        else:
            if query.is_flagged:
                query.is_flagged = False
                logger.info(f"Query {query_id} is no longer flagged. New performance score: {new_score:.2f}")

        self.db.commit()
        logger.info(f"Updated performance score for query {query_id} to {new_score:.2f}") 