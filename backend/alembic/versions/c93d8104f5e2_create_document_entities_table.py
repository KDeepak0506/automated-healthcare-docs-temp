"""create document entities table

Revision ID: c93d8104f5e2
Revises: b82c9103e4a1
Create Date: 2026-09-05 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'c93d8104f5e2'
down_revision: Union[str, Sequence[str], None] = 'b82c9103e4a1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'document_entities',
        sa.Column('entity_id', sa.UUID(), nullable=False, primary_key=True),
        sa.Column('document_id', sa.UUID(), sa.ForeignKey('documents.document_id', ondelete='CASCADE'), nullable=False),
        sa.Column('text', sa.String(), nullable=False),
        sa.Column('label', sa.String(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('start_char', sa.Integer(), nullable=True),
        sa.Column('end_char', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    op.create_index(op.f('ix_document_entities_document_id'), 'document_entities', ['document_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_document_entities_document_id'), table_name='document_entities')
    op.drop_table('document_entities')
