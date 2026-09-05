import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4
from sqlalchemy.orm import Session, sessionmaker

from app.models.document import Document
from app.models.document_text import DocumentText
from app.models.document_entity import DocumentEntity
from app.models.user import User
from app.services.ner_service import ClinicalNERService, clinical_ner_service
from app.services.document_service import _run_ocr_and_store, get_document_entities


# 1. Disease extraction
def test_disease_extraction():
    text = "Patient has Type 2 Diabetes and hypertension."
    entities = clinical_ner_service.extract_entities(text)
    labels = {e["text"].lower(): e["label"] for e in entities}
    assert any("diabetes" in t for t in labels)
    assert any("hypertension" in t for t in labels)


# 2. Medication extraction
def test_medication_extraction():
    text = "Patient is taking Metformin 500 mg twice daily."
    entities = clinical_ner_service.extract_entities(text)
    texts = [e["text"].lower() for e in entities]
    assert any("metformin" in t for t in texts)
    assert any("500 mg" in t for t in texts)


# 3. Laboratory test extraction
def test_lab_test_extraction():
    text = "HbA1c was 8.2%. Creatinine was 1.4 mg/dL."
    entities = clinical_ner_service.extract_entities(text)
    lab_tests = [e["text"] for e in entities if e["label"] == "lab test"]
    assert any("HbA1c" in t for t in lab_tests) or any("Creatinine" in t for t in lab_tests)


# 4. Laboratory value extraction
def test_lab_value_extraction():
    text = "HbA1c was 8.2%. Creatinine was 1.4 mg/dL."
    entities = clinical_ner_service.extract_entities(text)
    lab_vals = [e["text"] for e in entities if e["label"] == "lab value"]
    assert any("8.2%" in t or "8.2 %" in t for t in lab_vals)
    assert any("1.4 mg/dL" in t for t in lab_vals)


# 5. Procedure extraction
def test_procedure_extraction():
    text = "Patient underwent MRI."
    entities = clinical_ner_service.extract_entities(text)
    procedures = [e["text"] for e in entities if e["label"] == "procedure"]
    assert any("MRI" in p for p in procedures)


# 6. Symptom extraction
def test_symptom_extraction():
    text = "Patient reports fever and abdominal pain."
    entities = clinical_ner_service.extract_entities(text)
    symptoms = [e["text"].lower() for e in entities if e["label"] == "symptom"]
    assert any("fever" in s for s in symptoms) or any("abdominal pain" in s for s in symptoms)


# 7. OCR formatting
def test_ocr_formatting_extraction():
    text = "Glucose : 96 mg/dL\nCreatinine:1.4 mg / dL\nHbA1c : 8.2%"
    entities = clinical_ner_service.extract_entities(text)
    texts = [e["text"] for e in entities]
    assert any("96 mg/dL" in t for t in texts)
    assert any("1.4 mg" in t for t in texts)
    assert any("8.2%" in t or "8.2 %" in t for t in texts)


# 8. Duplicate removal
def test_duplicate_removal():
    text = "Microscopy: Pus cells. Microscopy: Pus cells."
    entities = clinical_ner_service.extract_entities(text)
    microscopy_ents = [e for e in entities if e["text"].lower() == "microscopy" and e["label"] == "procedure"]
    assert len(microscopy_ents) <= 1


# 9. Empty text
def test_empty_text_returns_empty_list():
    entities = clinical_ner_service.extract_entities("")
    assert entities == []
    entities_none = clinical_ner_service.extract_entities(None)
    assert entities_none == []


# 10. Non-clinical text
def test_non_clinical_text():
    text = "The quick brown fox jumps over the lazy dog in Seattle."
    entities = clinical_ner_service.extract_entities(text)
    assert len(entities) == 0


# 11. Privacy boundary (sanitized text only)
def test_privacy_boundary_sanitized_text_only(database: sessionmaker[Session], tmp_path):
    db = database()
    user_id = uuid4()
    doc_id = uuid4()

    doc = Document(
        document_id=doc_id,
        uploaded_by=user_id,
        file_name="patient_report.pdf",
        file_type="application/pdf",
        file_url=str(tmp_path / "patient_report.pdf"),
        processing_status="Pending",
        privacy_status="pending",
    )
    db.add(doc)
    db.commit()

    raw_ocr = "Patient John Doe (SSN 123-45-6789) diagnosed with Hypertension."
    
    with patch("app.services.document_service.extract_text") as mock_ocr, \
         patch("app.services.document_service.clinical_ner_service.extract_and_store_entities") as mock_ner:
        
        mock_ocr.return_value = MagicMock(
            raw_text=raw_ocr,
            page_count=1,
            confidence=0.95,
            layout={"pages": []},
            ocr_engine="pymupdf-native",
            processing_time_ms=150,
        )
        
        _run_ocr_and_store(doc_id, tmp_path / "patient_report.pdf", "application/pdf")
        
        mock_ner.assert_called_once()
        called_sanitized_text = mock_ner.call_args[1]["sanitized_text"]
        assert "123-45-6789" not in called_sanitized_text
        assert "John Doe" not in called_sanitized_text
        assert "Hypertension" in called_sanitized_text

    db.close()


# 12. Raw PHI protection
def test_raw_phi_protection(database: sessionmaker[Session], tmp_path):
    db = database()
    doc_id = uuid4()
    
    sanitized = "Patient [REDACTED] diagnosed with Hypertension. Prescribed Metformin 500 mg."
    clinical_ner_service.extract_and_store_entities(db, doc_id, sanitized)
    
    stored = db.query(DocumentEntity).filter(DocumentEntity.document_id == doc_id).all()
    for ent in stored:
        assert "[REDACTED]" not in ent.text
        assert "John Doe" not in ent.text
        assert "123-45-6789" not in ent.text

    db.close()


# 13. API authentication
def test_api_authentication(client: MagicMock):
    doc_id = uuid4()
    response = client.get(f"/api/v1/documents/{doc_id}/entities")
    assert response.status_code == 401


# 14. Ownership authorization
def test_api_ownership_authorization(client: MagicMock, database: sessionmaker[Session], auth_headers: dict):
    db = database()
    other_user_id = uuid4()
    doc_id = uuid4()
    
    doc = Document(
        document_id=doc_id,
        uploaded_by=other_user_id,
        file_name="other_user_doc.pdf",
        file_type="application/pdf",
        file_url="/tmp/other_user_doc.pdf",
        processing_status="Completed",
        privacy_status="completed",
    )
    db.add(doc)
    db.commit()
    db.close()

    response = client.get(f"/api/v1/documents/{doc_id}/entities", headers=auth_headers)
    assert response.status_code == 403


# 15. Privacy gating prevents M4 execution
def test_privacy_gating_prevents_m4_execution(database: sessionmaker[Session]):
    db = database()
    doc_id = uuid4()

    doc = Document(
        document_id=doc_id,
        uploaded_by=uuid4(),
        file_name="test.pdf",
        file_type="application/pdf",
        file_url="/tmp/test.pdf",
        processing_status="Failed",
        privacy_status="failed",
    )
    doc_text = DocumentText(
        document_id=doc_id,
        raw_text="CONFIDENTIAL PATIENT DATA - Hypertension",
        sanitized_text=None,
        page_count=1,
        ocr_engine="pymupdf-native",
    )
    db.add(doc)
    db.add(doc_text)
    db.commit()

    with patch("app.services.document_service.extract_text") as mock_ocr, \
         patch("app.services.document_service.clinical_ner_service.extract_and_store_entities") as mock_ner:
        
        mock_ocr.return_value = MagicMock(
            raw_text="CONFIDENTIAL PATIENT DATA - Hypertension",
            page_count=1,
            confidence=0.9,
            layout={"pages": []},
            ocr_engine="pytesseract",
            processing_time_ms=100,
        )

        with patch("app.services.document_service.privacy_service.sanitize", side_effect=Exception("Privacy error")):
            _run_ocr_and_store(doc_id, "test.pdf", "application/pdf")

    mock_ner.assert_not_called()
    db.close()


# 16. Realistic laboratory report test
def test_realistic_laboratory_report():
    sample_lab_report = """CLINICAL LABORATORY REPORT

PATIENT INFORMATION:
Patient Name: [REDACTED]
Age / Sex: 52 / M
Date: 2026-09-05

CLINICAL HISTORY & DIAGNOSIS:
Patient presented with severe hypertension and shortness of breath. History of Type 2 Diabetes Mellitus.
Currently taking Metformin 500 mg twice daily and Aspirin 81 mg daily.

PHYSICAL EXAMINATION & VITAL SIGNS:
Blood Pressure: 140/90 mmHg
Heart Rate: 78 bpm
Temperature: 98.6 °F

BIOCHEMICAL & LABORATORY EXAMINATION:
Fasting Blood Sugar / Glucose: 126 mg/dL
HbA1c / Glycated Hemoglobin: 8.2 %
Serum Creatinine: 1.4 mg/dL
Blood Urea Nitrogen (BUN): 18 mg/dL
Total Cholesterol: 215 mg/dL
HDL Cholesterol: 42 mg/dL
LDL Cholesterol: 138 mg/dL
Triglycerides: 175 mg/dL
Serum Sodium: 138 mEq/L
Serum Potassium: 4.2 mEq/L
Hemoglobin: 13.5 g/dL
WBC Count: 6.8 x 10^3 / uL
Platelet Count: 240,000 / uL

PROCEDURES & INVESTIGATIONS:
Patient underwent Electrocardiogram (ECG) and Chest X-Ray.
Microscopy: Pus cells 2-4 / hpf.
"""

    entities = clinical_ner_service.extract_entities(sample_lab_report)
    assert len(entities) >= 20  # Substantial entity recall improvement

    labels = {e["label"] for e in entities}
    assert "disease" in labels
    assert "medication" in labels
    assert "lab test" in labels
    assert "lab value" in labels
    assert "procedure" in labels


# 17. Precision Cleanup Test 1: Medication Dosage
def test_precision_medication_dosage():
    text = "Patient is taking Metformin 500 mg twice daily and Aspirin 81 mg daily."
    entities = clinical_ner_service.extract_entities(text)
    
    dosages = [e["text"] for e in entities if e["label"] == "dosage"]
    lab_vals = [e["text"] for e in entities if e["label"] == "lab value"]
    
    assert "500 mg" in dosages
    assert "81 mg" in dosages
    assert "500 mg" not in lab_vals
    assert "81 mg" not in lab_vals


# 18. Precision Cleanup Test 2: Laboratory Values (No Dosage Duplication)
def test_precision_laboratory_values():
    text = "Glucose: 192 mg/dL\nCreatinine: 1.41 mg/dL\nSodium: 139 mmol/L\neGFR: 57.10 mL/min"
    entities = clinical_ner_service.extract_entities(text)
    
    lab_tests = [e["text"] for e in entities if e["label"] == "lab test"]
    lab_vals = [e["text"] for e in entities if e["label"] == "lab value"]
    dosages = [e["text"] for e in entities if e["label"] == "dosage"]

    assert any("Glucose" in t for t in lab_tests)
    assert any("Creatinine" in t for t in lab_tests)
    assert any("Sodium" in t for t in lab_tests)
    assert any("eGFR" in t for t in lab_tests)

    assert "192 mg/dL" in lab_vals
    assert "1.41 mg/dL" in lab_vals
    assert "139 mmol/L" in lab_vals
    assert "57.10 mL/min" in lab_vals

    # Measurements MUST NOT appear as dosage
    assert len(dosages) == 0


# 19. Precision Cleanup Test 3: Mixed Medication + Lab Report
def test_precision_mixed_medication_and_lab():
    text = "Metformin 500 mg daily.\nCreatinine 1.41 mg/dL.\nGlucose 192 mg/dL."
    entities = clinical_ner_service.extract_entities(text)

    meds = [e["text"] for e in entities if e["label"] == "medication"]
    dosages = [e["text"] for e in entities if e["label"] == "dosage"]
    lab_tests = [e["text"] for e in entities if e["label"] == "lab test"]
    lab_vals = [e["text"] for e in entities if e["label"] == "lab value"]

    assert "Metformin" in meds
    assert "500 mg" in dosages
    assert any("Creatinine" in t for t in lab_tests)
    assert any("Glucose" in t for t in lab_tests)
    assert "1.41 mg/dL" in lab_vals
    assert "192 mg/dL" in lab_vals
    assert "1.41 mg" not in dosages
    assert "192 mg" not in dosages


# 20. Precision Cleanup Test 4: Long GLiNER Span Cleanup
def test_precision_long_gliner_span_cleanup():
    text = (
        "Decreased ESR is seen in some conditions.\n"
        "Factors That Interfere With Hba1c Measurement.\n"
        "The measured serum TSH concentrations may vary."
    )
    entities = clinical_ner_service.extract_entities(text)
    lab_tests = [e["text"] for e in entities if e["label"] == "lab test"]

    assert "ESR" in lab_tests
    assert any("hba1c" in t.lower() for t in lab_tests)
    assert "TSH" in lab_tests

    # Verify full sentences are NOT persisted as lab test entities
    assert "Decreased ESR is seen in some conditions." not in lab_tests
    assert "Factors That Interfere With Hba1c Measurement." not in lab_tests


# 21. Precision Cleanup Test 5: OCR Formatting
def test_precision_ocr_formatting():
    text = "Creatinine:\n1.41\nmg/dL"
    entities = clinical_ner_service.extract_entities(text)
    lab_tests = [e["text"] for e in entities if e["label"] == "lab test"]
    lab_vals = [e["text"] for e in entities if e["label"] == "lab value"]

    assert any("Creatinine" in t for t in lab_tests)
    assert any("1.41" in v for v in lab_vals)


# 22. Precision Cleanup Test 6: Duplicate Entities
def test_precision_duplicate_entities():
    text = "Microscopy: Pus cells 2-4 / hpf. Microscopy: Pus cells 2-4 / hpf."
    entities = clinical_ner_service.extract_entities(text)
    unique_pairs = {(e["text"].lower(), e["label"]) for e in entities}
    assert len(unique_pairs) == len(entities)
