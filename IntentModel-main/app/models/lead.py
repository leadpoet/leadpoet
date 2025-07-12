"""
Lead model for the leads table.
Implements the schema from BRD Section 7.
"""

from sqlalchemy import Column, String, DateTime, Boolean, JSON, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB, BIGINT
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid

from app.core.database import Base


class Lead(Base):
    """Lead model representing B2B leads."""
    
    __tablename__ = "leads"
    
    # Primary key
    lead_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Company information
    company_id = Column(String(255), nullable=False, index=True)
    company_name = Column(String(500), nullable=False)
    email = Column(String(255), nullable=False, index=True)
    
    # Firmographics (JSONB for flexible schema)
    firmographics = Column(JSON)
    
    # Technographics (JSONB for flexible schema)
    technographics = Column(JSON)
    
    # Intent data
    intent_score = Column(String(50), nullable=True)  # Current intent score
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    
    # Source tracking
    source = Column(String(100), nullable=True)
    source_id = Column(String(255), nullable=True)
    
    # Lead metadata
    lead_metadata = Column(JSONB, nullable=True, default=dict)
    
    # Simhash for plagiarism/similarity check
    simhash = Column(BIGINT, index=True)
    is_potential_duplicate = Column(Boolean, default=False)
    
    # Foreign Key to associate with a Query
    query_id = Column(UUID(as_uuid=True), ForeignKey("queries.id"), nullable=True)
    query = relationship("Query", back_populates="leads")
    
    # Relationship
    intent_snippets = relationship("IntentSnippet", back_populates="lead", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Lead(lead_id={self.lead_id}, company_name='{self.company_name}', email='{self.email}')>"
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'lead_id': str(self.lead_id),
            'company_id': self.company_id,
            'company_name': self.company_name,
            'email': self.email,
            'firmographics': self.firmographics or {},
            'technographics': self.technographics or {},
            'intent_score': self.intent_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'is_active': self.is_active,
            'source': self.source,
            'source_id': self.source_id,
            'lead_metadata': self.lead_metadata or {},
            'simhash': self.simhash,
            'is_potential_duplicate': self.is_potential_duplicate,
            'query_id': str(self.query_id) if self.query_id else None
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create Lead instance from dictionary."""
        # Validate required fields
        required_fields = ['company_id', 'company_name', 'email']
        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        # Handle lead_id if present
        lead_id = data.get('lead_id')
        if lead_id and isinstance(lead_id, str):
            try:
                lead_id = uuid.UUID(lead_id)
            except ValueError:
                lead_id = None

        # Handle datetime fields
        created_at = data.get('created_at')
        if created_at and isinstance(created_at, str):
            try:
                from datetime import datetime
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            except ValueError:
                created_at = None

        updated_at = data.get('updated_at')
        if updated_at and isinstance(updated_at, str):
            try:
                from datetime import datetime
                updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            except ValueError:
                updated_at = None

        instance = cls(
            company_id=data.get('company_id'),
            company_name=data.get('company_name'),
            email=data.get('email'),
            firmographics=data.get('firmographics', {}),
            technographics=data.get('technographics', {}),
            intent_score=data.get('intent_score'),
            is_active=data.get('is_active', True),
            source=data.get('source'),
            source_id=data.get('source_id'),
            lead_metadata=data.get('lead_metadata', {}),
            query_id=data.get('query_id'),
            simhash=data.get('simhash'),
            is_potential_duplicate=data.get('is_potential_duplicate', False)
        )
        
        # Set lead_id if provided
        if lead_id:
            instance.lead_id = lead_id
        
        # Set datetime fields if provided
        if created_at:
            instance.created_at = created_at
        if updated_at:
            instance.updated_at = updated_at
            
        return instance