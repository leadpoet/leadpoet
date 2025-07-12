"""
Intent Snippet model for the intent_snippets table.
Implements the schema from BRD Section 7 with composite PK.
"""

from sqlalchemy import Column, String, DateTime, Float, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid

from app.core.database import Base


class IntentSnippet(Base):
    """Intent snippet model for storing intent signals."""
    
    __tablename__ = "intent_snippets"
    
    # Composite primary key
    lead_id = Column(UUID(as_uuid=True), ForeignKey('leads.lead_id'), primary_key=True)
    snippet_id = Column(String(255), primary_key=True)
    
    # Snippet content
    content = Column(Text, nullable=False)
    content_type = Column(String(50), nullable=False)  # 'webpage', 'email', 'social', etc.
    
    # Scoring
    bm25_score = Column(Float, nullable=True)  # BM25 relevance score
    llm_score = Column(Float, nullable=True)   # LLM intent score
    
    # Source information
    source_url = Column(String(1000), nullable=True)
    source_domain = Column(String(255), nullable=True)
    source_type = Column(String(100), nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Snippet metadata
    snippet_metadata = Column(JSONB, nullable=True, default=dict)
    
    # Relationship
    lead = relationship("Lead", back_populates="intent_snippets")
    
    def __repr__(self):
        return f"<IntentSnippet(lead_id={self.lead_id}, snippet_id='{self.snippet_id}', content_type='{self.content_type}')>"
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'lead_id': str(self.lead_id),
            'snippet_id': self.snippet_id,
            'content': self.content,
            'content_type': self.content_type,
            'bm25_score': self.bm25_score,
            'llm_score': self.llm_score,
            'source_url': self.source_url,
            'source_domain': self.source_domain,
            'source_type': self.source_type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'snippet_metadata': self.snippet_metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create IntentSnippet instance from dictionary."""
        return cls(
            lead_id=data.get('lead_id'),
            snippet_id=data.get('snippet_id'),
            content=data.get('content'),
            content_type=data.get('content_type'),
            bm25_score=data.get('bm25_score'),
            llm_score=data.get('llm_score'),
            source_url=data.get('source_url'),
            source_domain=data.get('source_domain'),
            source_type=data.get('source_type'),
            snippet_metadata=data.get('snippet_metadata', {})
        ) 