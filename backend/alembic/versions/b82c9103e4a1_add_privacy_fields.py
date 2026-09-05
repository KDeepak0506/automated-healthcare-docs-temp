"""add privacy fields

Revision ID: b82c9103e4a1
Revises: aaaa723bebd8
Create Date: 2026-09-05 13:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b82c9103e4a1'
down_revision: Union[str, Sequence[str], None] = 'aaaa723bebd8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'documents',
        sa.Column('privacy_status', sa.String(), nullable=False, server_default='pending'),
    )
    op.add_column(
        'document_text',
        sa.Column('sanitized_text', sa.Text(), nullable=True),
    )
    op.add_column(
        'document_text',
        sa.Column('privacy_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('document_text', 'privacy_metadata')
    op.drop_column('document_text', 'sanitized_text')
    op.drop_column('documents', 'privacy_status')
