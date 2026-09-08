"""Tests for Patient domain: CRUD, access control, and patient assignment."""
from datetime import date
from typing import Generator
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.models.user import User
from app.schemas.user import UserRole
from app.schemas.patient import PatientCreate
from app.services.patient_service import (
    assign_patient_to_user,
    create_patient,
    has_patient_access,
    list_accessible_patients,
    verify_patient_access,
)


def _make_user(db: Session, email: str, role: str) -> User:
    user = User(
        name=f"User {email}",
        email=email,
        password_hash="hashed",
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_patient_creation_and_creator_access(database: sessionmaker[Session]) -> None:
    with database() as db:
        doctor = _make_user(db, "creator.doctor@test.org", UserRole.DOCTOR.value)

        patient = create_patient(
            db,
            PatientCreate(
                mrn="PAT-99001",
                first_name="Alice",
                last_name="Johnson",
                date_of_birth=date(1985, 5, 20),
                gender="Female",
            ),
            creator=doctor,
        )

        assert patient.patient_id is not None
        assert patient.mrn == "PAT-99001"
        assert patient.first_name == "Alice"
        # Creator gets automatic access
        assert has_patient_access(db, doctor, patient.patient_id) is True


def test_unassigned_user_cannot_access_patient(database: sessionmaker[Session]) -> None:
    with database() as db:
        creator = _make_user(db, "creator2@test.org", UserRole.DOCTOR.value)
        other = _make_user(db, "other2@test.org", UserRole.DOCTOR.value)

        patient = create_patient(
            db,
            PatientCreate(mrn="PAT-99002", first_name="Bob", last_name="Smith"),
            creator=creator,
        )

        assert has_patient_access(db, other, patient.patient_id) is False


def test_admin_always_has_patient_access(database: sessionmaker[Session]) -> None:
    with database() as db:
        creator = _make_user(db, "creator3@test.org", UserRole.DOCTOR.value)
        admin = _make_user(db, "admin3@test.org", UserRole.ADMIN.value)

        patient = create_patient(
            db,
            PatientCreate(mrn="PAT-99003", first_name="Carol", last_name="White"),
            creator=creator,
        )

        assert has_patient_access(db, admin, patient.patient_id) is True


def test_assign_grants_patient_access(database: sessionmaker[Session]) -> None:
    with database() as db:
        creator = _make_user(db, "creator4@test.org", UserRole.DOCTOR.value)
        admin = _make_user(db, "admin4@test.org", UserRole.ADMIN.value)
        other = _make_user(db, "other4@test.org", UserRole.NURSE.value)

        patient = create_patient(
            db,
            PatientCreate(mrn="PAT-99004", first_name="Dave", last_name="Green"),
            creator=creator,
        )

        # Before assignment
        assert has_patient_access(db, other, patient.patient_id) is False

        # Assign
        assign_patient_to_user(db, patient.patient_id, other.user_id, assigner=admin)

        # After assignment
        assert has_patient_access(db, other, patient.patient_id) is True


def test_duplicate_mrn_rejected(database: sessionmaker[Session]) -> None:
    with database() as db:
        creator = _make_user(db, "creator5@test.org", UserRole.RECORDS_STAFF.value)

        create_patient(db, PatientCreate(mrn="PAT-DUP", first_name="X", last_name="Y"), creator)

        with pytest.raises(HTTPException) as exc_info:
            create_patient(db, PatientCreate(mrn="PAT-DUP", first_name="A", last_name="B"), creator)
        assert exc_info.value.status_code == 400


def test_list_accessible_patients_scoped(database: sessionmaker[Session]) -> None:
    with database() as db:
        doctor = _make_user(db, "lister@test.org", UserRole.DOCTOR.value)
        other = _make_user(db, "other_lister@test.org", UserRole.DOCTOR.value)

        create_patient(db, PatientCreate(mrn="LP-001", first_name="P1", last_name="Last"), creator=doctor)
        create_patient(db, PatientCreate(mrn="LP-002", first_name="P2", last_name="Last"), creator=doctor)

        result = list_accessible_patients(db, user=doctor)
        assert result["total"] == 2

        # Other doctor should see no patients
        result_other = list_accessible_patients(db, user=other)
        assert result_other["total"] == 0


def test_verify_patient_access_raises_on_unauthorized(database: sessionmaker[Session]) -> None:
    with database() as db:
        creator = _make_user(db, "owner6@test.org", UserRole.DOCTOR.value)
        intruder = _make_user(db, "intruder6@test.org", UserRole.DOCTOR.value)

        patient = create_patient(
            db,
            PatientCreate(mrn="PAT-99006", first_name="Ed", last_name="Brown"),
            creator=creator,
        )

        with pytest.raises(HTTPException) as exc_info:
            verify_patient_access(db, user=intruder, patient_id=patient.patient_id)
        assert exc_info.value.status_code == 403


def test_records_staff_sees_all_patients(database: sessionmaker[Session]) -> None:
    with database() as db:
        doctor = _make_user(db, "doc7@test.org", UserRole.DOCTOR.value)
        staff = _make_user(db, "staff7@test.org", UserRole.RECORDS_STAFF.value)

        create_patient(db, PatientCreate(mrn="RS-001", first_name="F1", last_name="L1"), creator=doctor)
        create_patient(db, PatientCreate(mrn="RS-002", first_name="F2", last_name="L2"), creator=doctor)

        result = list_accessible_patients(db, user=staff)
        assert result["total"] == 2
