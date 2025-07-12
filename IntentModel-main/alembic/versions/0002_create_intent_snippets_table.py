"""Create intent_snippets table

Revision ID: 0002
Revises: 0001
Create Date: 2025-01-27 10:01:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create intent_snippets table
    op.create_table('intent_snippets',
        sa.Column('lead_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('snippet_id', sa.String(length=255), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('content_type', sa.String(length=50), nullable=False),
        sa.Column('bm25_score', sa.Float(), nullable=True),
        sa.Column('llm_score', sa.Float(), nullable=True),
        sa.Column('source_url', sa.String(length=1000), nullable=True),
        sa.Column('source_domain', sa.String(length=255), nullable=True),
        sa.Column('source_type', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['lead_id'], ['leads.lead_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('lead_id', 'snippet_id')
    )
    
    # Create indexes
    op.create_index(op.f('ix_intent_snippets_content_type'), 'intent_snippets', ['content_type'], unique=False)
    op.create_index(op.f('ix_intent_snippets_source_domain'), 'intent_snippets', ['source_domain'], unique=False)
    op.create_index(op.f('ix_intent_snippets_created_at'), 'intent_snippets', ['created_at'], unique=False)
    op.create_index(op.f('ix_intent_snippets_bm25_score'), 'intent_snippets', ['bm25_score'], unique=False)
    op.create_index(op.f('ix_intent_snippets_llm_score'), 'intent_snippets', ['llm_score'], unique=False)
    
    # Create GIN index for JSONB metadata column
    op.execute('CREATE INDEX ix_intent_snippets_metadata ON intent_snippets USING GIN (metadata)')


def downgrade() -> None:
    # Drop indexes
    op.drop_index(op.f('ix_intent_snippets_metadata'), table_name='intent_snippets')
    op.drop_index(op.f('ix_intent_snippets_llm_score'), table_name='intent_snippets')
    op.drop_index(op.f('ix_intent_snippets_bm25_score'), table_name='intent_snippets')
    op.drop_index(op.f('ix_intent_snippets_created_at'), table_name='intent_snippets')
    op.drop_index(op.f('ix_intent_snippets_source_domain'), table_name='intent_snippets')
    op.drop_index(op.f('ix_intent_snippets_content_type'), table_name='intent_snippets')
    
    # Drop table
    op.drop_table('intent_snippets') 