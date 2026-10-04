from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, require_roles
from app.db.session import get_db
from app.models.user import User
from app.schemas.user import UserResponse, UserRole

router = APIRouter(
    prefix="/users",
    tags=["Users"],
)


@router.get(
    "",
    response_model=list[UserResponse],
    dependencies=[Depends(require_roles(UserRole.ADMIN.value, UserRole.RECORDS_STAFF.value))],
)
def list_users_endpoint(
    role: str | None = Query(default=None, description="Filter users by role"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List users optionally filtered by role. Restricted to Admin or Records Staff."""
    query = db.query(User)
    if role is not None:
        role_val = role.value if hasattr(role, "value") else str(role).strip().lower()
        query = query.filter(User.role == role_val)

    return query.order_by(User.name.asc()).all()
