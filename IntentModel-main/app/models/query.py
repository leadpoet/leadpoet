import uuid
from sqlalchemy import Column, String, DateTime, Float, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.core.database import Base
import datetime

class Query(Base):
    __tablename__ = "queries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index=True) 
    query_text = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Miner Quality Factor (MQF) fields
    mqf_score = Column(Float, default=0.0)
    is_flagged = Column(Boolean, default=False)
    
    leads = relationship("Lead", back_populates="query") 