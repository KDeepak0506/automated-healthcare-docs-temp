import pytest
from app.services.privacy_service import privacy_service


def test_sanitize_patient_identifiers_and_clinical_preservation():
    raw_text = """Patient Name: Rahul Kumar
DOB: 14/05/1998
MRN: MRN12345
UHID: UHID998877
Patient ID: PAT-5544
Insurance ID: INS-112233
Policy Number: POL-778899
Phone: 9876543210
Email: rahul@example.com
Address: 123 Healthcare Ave, Cityville
IP Number: IP-9988
Account Number: ACC-1122

HbA1c: 8.2%
Creatinine: 1.4 mg/dL
Diagnosis: Type 2 Diabetes
Medication: Metformin 500 mg
"""

    result = privacy_service.sanitize(raw_text)

    # Identifiers must be sanitized
    assert "Rahul Kumar" not in result.sanitized_text
    assert "14/05/1998" not in result.sanitized_text
    assert "MRN12345" not in result.sanitized_text
    assert "UHID998877" not in result.sanitized_text
    assert "PAT-5544" not in result.sanitized_text
    assert "INS-112233" not in result.sanitized_text
    assert "POL-778899" not in result.sanitized_text
    assert "9876543210" not in result.sanitized_text
    assert "rahul@example.com" not in result.sanitized_text

    # Tags must be present
    assert "[PATIENT]" in result.sanitized_text
    assert "[DATE]" in result.sanitized_text
    assert "[MRN]" in result.sanitized_text
    assert "[PHONE]" in result.sanitized_text
    assert "[EMAIL]" in result.sanitized_text

    # Clinical data must be preserved
    assert "HbA1c: 8.2%" in result.sanitized_text
    assert "Creatinine: 1.4 mg/dL" in result.sanitized_text
    assert "Diagnosis: Type 2 Diabetes" in result.sanitized_text
    assert "Medication: Metformin 500 mg" in result.sanitized_text


def test_sanitize_empty_text():
    result = privacy_service.sanitize("")
    assert result.sanitized_text == ""
    assert result.entities_detected == 0

    result_none = privacy_service.sanitize(None)
    assert result_none.sanitized_text == ""
    assert result_none.entities_detected == 0


def test_sanitize_multiple_identifiers():
    sample = "Patient Name: Jane Doe. Phone: 555-123-4567. DOB: 01/01/1990."
    res = privacy_service.sanitize(sample)
    assert "Jane Doe" not in res.sanitized_text
    assert "555-123-4567" not in res.sanitized_text
    assert "01/01/1990" not in res.sanitized_text
    assert "[PATIENT]" in res.sanitized_text
    assert "[PHONE]" in res.sanitized_text
    assert "[DATE]" in res.sanitized_text
