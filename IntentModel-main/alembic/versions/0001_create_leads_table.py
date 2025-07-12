"""Create leads table

Revision ID: 0001
Revises: 
Create Date: 2025-01-27 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create leads table
    op.create_table('leads',
        sa.Column('lead_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('company_id', sa.String(length=255), nullable=False),
        sa.Column('company_name', sa.String(length=500), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('firmographics', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('technographics', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('intent_snippets', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('intent_score', sa.Float(), nullable=True),        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('source', sa.String(length=100), nullable=True),
        sa.Column('source_id', sa.String(length=255), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('lead_id')
    )
    
    # Create indexes
    op.create_index(op.f('ix_leads_company_id'), 'leads', ['company_id'], unique=False)
    op.create_index(op.f('ix_leads_email'), 'leads', ['email'], unique=False)
    op.create_index(op.f('ix_leads_is_active'), 'leads', ['is_active'], unique=False)
    op.create_index(op.f('ix_leads_created_at'), 'leads', ['created_at'], unique=False)
    
    # Create GIN indexes for JSONB columns
    op.execute('CREATE INDEX ix_leads_firmographics ON leads USING GIN (firmographics)')
    op.execute('CREATE INDEX ix_leads_technographics ON leads USING GIN (technographics)')
    op.execute('CREATE INDEX ix_leads_metadata ON leads USING GIN (metadata)')


def downgrade() -> None:
    # Drop indexes
    op.drop_index(op.f('ix_leads_metadata'), table_name='leads')
    op.drop_index(op.f('ix_leads_technographics'), table_name='leads')
    op.drop_index(op.f('ix_leads_firmographics'), table_name='leads')
    op.drop_index(op.f('ix_leads_created_at'), table_name='leads')
    op.drop_index(op.f('ix_leads_is_active'), table_name='leads')
    op.drop_index(op.f('ix_leads_email'), table_name='leads')
    op.drop_index(op.f('ix_leads_company_id'), table_name='leads')
    
    # Drop table
    op.drop_table('leads') 