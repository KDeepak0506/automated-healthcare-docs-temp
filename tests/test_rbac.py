"""Tests for role-based access control: registration privilege guard."""
import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session, sessionmaker

from app.schemas.user import UserCreate, UserRole
from app.services.auth_service import register_user


def test_public_registration_admin_blocked(database: sessionmaker[Session]) -> None:
    """Public registration must reject admin role assignment."""
    with database() as db:
        user_in = UserCreate(
            name="Hacker Admin",
            email="hacker@hospital.org",
            password="password123",
            role=UserRole.ADMIN,
        )
        with pytest.raises(HTTPException) as exc_info:
            register_user(db, user_in)

        assert exc_info.value.status_code == 400
        assert "administrator" in exc_info.value.detail.lower()


def test_public_registration_doctor_allowed(database: sessionmaker[Session]) -> None:
    """Public registration must allow doctor role."""
    with database() as db:
        user_in = UserCreate(
            name="Valid Doctor",
            email="valid.doctor@hospital.org",
            password="password123",
            role=UserRole.DOCTOR,
        )
        user = register_user(db, user_in)
        assert user.user_id is not None
        assert user.role == UserRole.DOCTOR.value


def test_public_registration_nurse_allowed(database: sessionmaker[Session]) -> None:
    """Public registration must allow nurse role."""
    with database() as db:
        user_in = UserCreate(
            name="Nurse Jane",
            email="nurse.jane@hospital.org",
            password="password123",
            role=UserRole.NURSE,
        )
        user = register_user(db, user_in)
        assert user.role == UserRole.NURSE.value


def test_public_registration_records_staff_allowed(database: sessionmaker[Session]) -> None:
    """Public registration must allow records_staff role."""
    with database() as db:
        user_in = UserCreate(
            name="Records Clerk",
            email="records.clerk@hospital.org",
            password="password123",
            role=UserRole.RECORDS_STAFF,
        )
        user = register_user(db, user_in)
        assert user.role == UserRole.RECORDS_STAFF.value


def test_admin_registration_via_api(client) -> None:
    """API endpoint must reject admin role during public registration."""
    res = client.post(
        "/api/v1/auth/register",
        json={
            "name": "API Admin Attacker",
            "email": "api.admin@hospital.org",
            "password": "password123",
            "role": "admin",
        },
    )
    assert res.status_code == 400
    assert "administrator" in res.json()["detail"].lower()
