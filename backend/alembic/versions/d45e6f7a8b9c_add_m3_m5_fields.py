"""add m3 and m5 fields

Revision ID: d45e6f7a8b9c
Revises: c93d8104f5e2
Create Date: 2026-09-05 22:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd45e6f7a8b9c'
down_revision: Union[str, Sequence[str], None] = 'c93d8104f5e2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'documents',
        sa.Column('classification_confidence', sa.Float(), nullable=True),
    )
    op.add_column(
        'documents',
        sa.Column('summary', sa.Text(), nullable=True),
    )
    op.add_column(
        'documents',
        sa.Column('key_findings', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('documents', 'key_findings')
    op.drop_column('documents', 'summary')
    op.drop_column('documents', 'classification_confidence')
