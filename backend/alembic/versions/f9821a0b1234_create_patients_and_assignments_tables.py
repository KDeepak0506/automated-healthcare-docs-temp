"""create patients and assignments tables and link documents.patient_id

Revision ID: f9821a0b1234
Revises: e56f7a8b9c0d
Create Date: 2026-09-08 16:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import text

# revision identifiers, used by Alembic.
revision: str = 'f9821a0b1234'
down_revision: Union[str, Sequence[str], None] = 'e56f7a8b9c0d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create patients table
    op.create_table(
        'patients',
        sa.Column('patient_id', sa.UUID(), nullable=False),
        sa.Column('mrn', sa.String(), nullable=False),
        sa.Column('first_name', sa.String(), nullable=False),
        sa.Column('last_name', sa.String(), nullable=False),
        sa.Column('date_of_birth', sa.Date(), nullable=True),
        sa.Column('gender', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('patient_id')
    )
    op.create_index(op.f('ix_patients_mrn'), 'patients', ['mrn'], unique=True)

    # 2. Create user_patient_assignments table
    op.create_table(
        'user_patient_assignments',
        sa.Column('assignment_id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('patient_id', sa.UUID(), nullable=False),
        sa.Column('assigned_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.patient_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('assignment_id')
    )
    op.create_index(op.f('ix_user_patient_assignments_patient_id'), 'user_patient_assignments', ['patient_id'], unique=False)
    op.create_index(op.f('ix_user_patient_assignments_user_id'), 'user_patient_assignments', ['user_id'], unique=False)

    # 3. Safe Migration Strategy for existing documents.patient_id values
    bind = op.get_bind()
    conn = bind

    # Check if documents table exists and has patient_id column
    result = conn.execute(text("SELECT patient_id FROM documents WHERE patient_id IS NOT NULL"))
    existing_patient_ids = set()
    for row in result:
        if row[0]:
            existing_patient_ids.add(str(row[0]))

    for p_id in existing_patient_ids:
        short_id = p_id[:8].upper()
        conn.execute(
            text(
                "INSERT INTO patients (patient_id, mrn, first_name, last_name, created_at, updated_at) "
                "VALUES (:pid, :mrn, 'Patient', :last_name, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
                "ON CONFLICT (patient_id) DO NOTHING"
            ),
            {"pid": p_id, "mrn": f"PAT-MIGRATED-{short_id}", "last_name": f"Record ({short_id})"}
        )

    # 4. Add FK constraint on documents.patient_id
    op.create_foreign_key(
        'fk_documents_patient_id_patients',
        'documents', 'patients',
        ['patient_id'], ['patient_id'],
        ondelete='SET NULL'
    )


def downgrade() -> None:
    op.drop_constraint('fk_documents_patient_id_patients', 'documents', type_='foreignkey')
    op.drop_index(op.f('ix_user_patient_assignments_user_id'), table_name='user_patient_assignments')
    op.drop_index(op.f('ix_user_patient_assignments_patient_id'), table_name='user_patient_assignments')
    op.drop_table('user_patient_assignments')
    op.drop_index(op.f('ix_patients_mrn'), table_name='patients')
    op.drop_table('patients')
