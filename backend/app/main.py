from fastapi import FastAPI
from sqlalchemy import text

from app.db.session import engine
from app.routers import auth, document, patient, user


app = FastAPI(
    title="Healthcare Document Intelligence API",
    version="1.0.0",
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request
from app.services.patient_identity_service import PatientIdentityMismatchError

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(PatientIdentityMismatchError)
async def patient_identity_mismatch_handler(request: Request, exc: PatientIdentityMismatchError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.code,
            "message": exc.error_message,
            "detail": exc.error_message,
            "detected_patient_name": exc.detected_patient_name,
            "target_patient_name": exc.target_patient_name,
            "detected_mrn": exc.detected_mrn,
            "target_mrn": exc.target_mrn,
        },
    )


@app.get("/api/v1/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/v1/health/db")
def database_health_check():
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    return {"database": "ok"}


app.include_router(
    auth.router,
    prefix="/api/v1",
)

app.include_router(
    patient.router,
    prefix="/api/v1",
)

app.include_router(
    document.router,
    prefix="/api/v1",
)

app.include_router(
    user.router,
    prefix="/api/v1",
)
