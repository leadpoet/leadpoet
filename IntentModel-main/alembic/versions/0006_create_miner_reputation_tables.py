"""Create miner reputation tables

Revision ID: 0006
Revises: 0005
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade():
    # Create miner_reputation table
    op.create_table('miner_reputation',
        sa.Column('miner_id', sa.String(length=255), nullable=False),
        sa.Column('reputation_score', sa.Float(), nullable=False, default=1.0),
        sa.Column('pass_rate', sa.Float(), nullable=False, default=1.0),
        sa.Column('flag_rate', sa.Float(), nullable=False, default=0.0),
        sa.Column('total_queries', sa.Integer(), nullable=False, default=0),
        sa.Column('passed_queries', sa.Integer(), nullable=False, default=0),
        sa.Column('flagged_queries', sa.Integer(), nullable=False, default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_calculation_time', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('calculation_window_days', sa.Integer(), nullable=False, default=30),
        sa.PrimaryKeyConstraint('miner_id')
    )
    op.create_index(op.f('ix_miner_reputation_miner_id'), 'miner_reputation', ['miner_id'], unique=False)
    
    # Create miner_throttles table
    op.create_table('miner_throttles',
        sa.Column('miner_id', sa.String(length=255), nullable=False),
        sa.Column('is_throttled', sa.Boolean(), nullable=False, default=False),
        sa.Column('throttle_start_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('throttle_end_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('throttle_reason', sa.String(length=255), nullable=True),
        sa.Column('flag_rate_at_throttle', sa.Float(), nullable=True),
        sa.Column('reputation_score_at_throttle', sa.Float(), nullable=True),
        sa.Column('throttle_count', sa.Integer(), nullable=False, default=0),
        sa.Column('total_throttle_duration_hours', sa.Float(), nullable=False, default=0.0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('miner_id')
    )
    op.create_index(op.f('ix_miner_throttles_miner_id'), 'miner_throttles', ['miner_id'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_miner_throttles_miner_id'), table_name='miner_throttles')
    op.drop_table('miner_throttles')
    op.drop_index(op.f('ix_miner_reputation_miner_id'), table_name='miner_reputation')
    op.drop_table('miner_reputation') 