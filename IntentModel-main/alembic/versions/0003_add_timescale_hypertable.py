"""Add TimescaleDB hypertable partitioning

Revision ID: 0003
Revises: 0002
Create Date: 2025-01-27 10:02:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable TimescaleDB extension if not already enabled
    op.execute('CREATE EXTENSION IF NOT EXISTS timescaledb')
    
    # Convert leads table to hypertable with partitioning on created_at
    op.execute("""
        SELECT create_hypertable(
            'leads', 
            'created_at',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        )
    """)
    
    # Add compression policy (compress chunks older than 7 days)
    op.execute("""
        SELECT add_compression_policy(
            'leads',
            compress_after => INTERVAL '7 days',
            if_not_exists => TRUE
        )
    """)
    
    # Add retention policy (drop chunks older than 90 days)
    op.execute("""
        SELECT add_retention_policy(
            'leads',
            drop_after => INTERVAL '90 days',
            if_not_exists => TRUE
        )
    """)
    
    # Convert intent_snippets table to hypertable
    op.execute("""
        SELECT create_hypertable(
            'intent_snippets', 
            'created_at',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        )
    """)
    
    # Add compression policy for intent_snippets
    op.execute("""
        SELECT add_compression_policy(
            'intent_snippets',
            compress_after => INTERVAL '7 days',
            if_not_exists => TRUE
        )
    """)
    
    # Add retention policy for intent_snippets
    op.execute("""
        SELECT add_retention_policy(
            'intent_snippets',
            drop_after => INTERVAL '90 days',
            if_not_exists => TRUE
        )
    """)


def downgrade() -> None:
    # Remove retention policies
    op.execute("SELECT remove_retention_policy('intent_snippets', if_exists => TRUE)")
    op.execute("SELECT remove_retention_policy('leads', if_exists => TRUE)")
    
    # Remove compression policies
    op.execute("SELECT remove_compression_policy('intent_snippets', if_exists => TRUE)")
    op.execute("SELECT remove_compression_policy('leads', if_exists => TRUE)")
    
    # Convert hypertables back to regular tables
    op.execute("SELECT drop_hypertable('intent_snippets', if_exists => TRUE)")
    op.execute("SELECT drop_hypertable('leads', if_exists => TRUE)") 