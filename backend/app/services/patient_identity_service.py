import logging
import re
from dataclasses import dataclass
from pathlib import Path
from fastapi import HTTPException, status

from app.models.patient import Patient

logger = logging.getLogger(__name__)


class PatientIdentityMismatchError(HTTPException):
    """Exception raised when an uploaded document clearly belongs to a different patient."""

    def __init__(
        self,
        detected_patient_name: str | None = None,
        target_patient_name: str | None = None,
        message: str | None = None,
        detected_mrn: str | None = None,
        target_mrn: str | None = None,
    ):
        if not message:
            if detected_patient_name and target_patient_name:
                message = (
                    f"Document appears to belong to {detected_patient_name}, "
                    f"but this workspace is for {target_patient_name}. The document was not uploaded."
                )
            elif detected_mrn and target_mrn:
                message = (
                    f"Document appears to belong to patient with MRN '{detected_mrn}', "
                    f"but this workspace is for MRN '{target_mrn}'. The document was not uploaded."
                )
            else:
                message = "The uploaded document appears to belong to another patient."

        detail = {
            "code": "PATIENT_IDENTITY_MISMATCH",
            "message": message,
            "detected_patient_name": detected_patient_name,
            "target_patient_name": target_patient_name,
            "detected_mrn": detected_mrn,
            "target_mrn": target_mrn,
        }
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )
        self.code = "PATIENT_IDENTITY_MISMATCH"
        self.error_message = message
        self.detected_patient_name = detected_patient_name
        self.target_patient_name = target_patient_name
        self.detected_mrn = detected_mrn
        self.target_mrn = target_mrn


@dataclass
class DocumentPatientIdentity:
    patient_name: str | None = None
    mrn: str | None = None
    dob: str | None = None
    source: str = "none"  # "content", "filename", "none"


# Common titles and clinician designations to filter out
CLINICIAN_PREFIXES = {
    "dr.", "dr", "doctor", "md", "d.o.", "do", "r.n.", "rn", "n.p.", "np", "pa-c", "pa"
}
CLINICIAN_KEYWORDS = {
    "physician", "doctor", "surgeon", "attending", "consultant", "provider",
    "referrer", "referring", "pathologist", "radiologist", "fellow", "resident",
    "hospital", "clinic", "center", "health", "department", "laboratory", "dept", "lab",
    "nurse", "nursing", "clinician", "specialist", "therapist", "paramedic", "staff"
}
GENERIC_FILE_WORDS = {
    "report", "scan", "medical", "lab", "labs", "document", "doc", "results", "summary",
    "general", "note", "notes", "prescription", "patient", "image", "test", "tests", "file",
    "ct", "mri", "xray", "ultrasound", "blood", "panel", "discharge", "vital", "vitals",
    "sign", "signs", "chart", "charts", "record", "records", "intake", "order", "orders",
    "history", "exam", "examination", "card", "id", "form", "forms", "evaluation", "assessment",
    "plan", "encounter", "visit", "triage", "progress", "consult", "consultation", "referral",
    "medication", "meds", "rx", "care", "clinical", "sheet", "log", "slip", "case", "study",
    "screening", "checkup", "admission", "treatment", "therapy", "physio", "rehab", "sample",
    "specimen", "data", "info", "information"
}
STOP_WORDS = {
    "for", "a", "an", "the", "and", "or", "in", "on", "at", "to", "from", "by",
    "with", "of", "as", "is", "it", "my", "our", "your", "their", "her", "his",
    "new", "old", "copy", "final", "draft", "v1", "v2", "part", "page", "upload",
    "patient", "doc", "file"
}


def _clean_name(raw_name: str) -> str:
    """Clean extracted patient name candidate."""
    # Strip any trailing metadata tags like DOB, MRN, Age, Gender
    cleaned = re.split(r"(?i)\s+(?:DOB|Date\s*of\s*Birth|Age|Gender|Sex|MRN|UHID|Patient\s*ID|Date|Phone|Address)\b", raw_name)[0]
    # Remove characters like brackets, quotes, colons
    cleaned = re.sub(r"[\[\]{}():*#\"_]", " ", cleaned)
    # Remove honorifics if at start
    cleaned = re.sub(r"(?i)^(?:Mr\.?|Mrs\.?|Ms\.?|Miss)\s+", "", cleaned.strip())
    # Normalize multiple whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_clinician_or_facility(name: str) -> bool:
    """Check if the extracted name refers to a doctor, clinician, or medical facility."""
    lowered = name.lower()
    words = set(re.findall(r"\b[a-zA-Z]+\b", lowered))
    if words & CLINICIAN_KEYWORDS:
        return True
    first_word = lowered.split()[0] if lowered.split() else ""
    if first_word in CLINICIAN_PREFIXES:
        return True
    last_word = lowered.split()[-1] if lowered.split() else ""
    if last_word in {"md", "do", "rn", "np", "pa"}:
        return True
    return False


def _normalize_tokens(text: str) -> set[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return set(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))


def is_name_match(detected_name: str, target_first: str, target_last: str) -> bool:
    """
    Check if detected name matches the target patient's name.
    Handles 'First Last', 'Last, First', middle names, and initials.
    """
    det_lower = detected_name.lower().replace(",", " ")
    det_tokens = _normalize_tokens(det_lower)

    t_first = target_first.lower()
    t_last = target_last.lower()

    # Exact full name match in sequence
    full_target = f"{t_first} {t_last}"
    rev_target = f"{t_last} {t_first}"
    if full_target in det_lower or rev_target in det_lower:
        return True

    # Both first and last name present as words
    if t_first in det_tokens and t_last in det_tokens:
        return True

    # Last name match with first initial
    if t_last in det_tokens and any(tok.startswith(t_first[0]) for tok in det_tokens if len(tok) == 1 or tok == t_first):
        return True

    return False


def extract_patient_identity(raw_text: str, filename: str | None = None) -> DocumentPatientIdentity:
    """
    Extract patient-identifying information (name, MRN, DOB) from document text and filename.
    Prioritizes explicit patient headers and content over filename.
    """
    identity = DocumentPatientIdentity()

    if not raw_text and not filename:
        return identity

    text = raw_text or ""

    # 1. MRN / Patient ID extraction
    # Patterns: MRN: 12345, Medical Record Number: 12345, UHID: 12345, Patient ID: 12345
    mrn_match = re.search(
        r"(?i)\b(?:MRN|Medical\s*Record\s*(?:No|Number|#)?|UHID|Patient\s*ID|Pat\s*ID)\s*[:#\-–]?\s*([A-Za-z0-9-]+)",
        text,
    )
    if mrn_match:
        cand_mrn = mrn_match.group(1).strip()
        # Ensure it's not a generic placeholder like [MRN]
        if cand_mrn and not cand_mrn.startswith("[") and len(cand_mrn) >= 2:
            identity.mrn = cand_mrn
            identity.source = "content"

    # 2. Date of birth extraction
    dob_match = re.search(
        r"(?i)\b(?:DOB|Date\s*of\s*Birth|Birth\s*Date)\s*[:#\-–]?\s*([A-Za-z0-9,/\s-]+)",
        text,
    )
    if dob_match:
        cand_dob = dob_match.group(1).strip().split("\n")[0]
        if cand_dob and not cand_dob.startswith("[") and any(ch.isdigit() for ch in cand_dob):
            identity.dob = cand_dob

    # 3. Patient Name extraction from content
    # Look for explicit patient headers: "Patient Name:", "Patient:", "Pt Name:", "**Patient Name:**"
    explicit_name_pattern = re.compile(
        r"(?i)(?:\*\*Patient(?:\s+Name)?:\*\*|Patient\s*Name\s*[:#\-–]|Pt\.?\s*Name\s*[:#\-–]|Patient\s*[:#\-–])\s*([A-Za-z][A-Za-z\s.'-]{1,50})"
    )
    for m in explicit_name_pattern.finditer(text):
        cand = _clean_name(m.group(1))
        if cand and not cand.startswith("[") and len(cand) >= 3 and not _is_clinician_or_facility(cand):
            tokens = _normalize_tokens(cand)
            if not tokens.issubset(GENERIC_FILE_WORDS):
                identity.patient_name = cand
                identity.source = "content"
                break

    # If not found yet, check generic Name: header while filtering out prefixes like Doctor Name, Facility Name, etc.
    if not identity.patient_name:
        generic_name_pattern = re.compile(
            r"(?i)(?:([A-Za-z]+)\s+)?Name\s*[:#\-–]\s*([A-Za-z][A-Za-z\s.'-]{1,50})"
        )
        non_patient_prefixes = {
            "doctor", "physician", "facility", "hospital", "clinic", "provider",
            "nurse", "surgeon", "file", "test", "user", "lab", "department"
        }
        for m in generic_name_pattern.finditer(text):
            prefix = (m.group(1) or "").lower()
            if prefix in non_patient_prefixes:
                continue
            cand = _clean_name(m.group(2))
            if cand and not cand.startswith("[") and len(cand) >= 3 and not _is_clinician_or_facility(cand):
                tokens = _normalize_tokens(cand)
                if not tokens.issubset(GENERIC_FILE_WORDS):
                    identity.patient_name = cand
                    identity.source = "content"
                    break
    # If content has an explicit patient name, return immediately
    if identity.patient_name:
        return identity

    # 4. If no explicit patient name was found in content, check filename as an additional signal
    if filename:
        stem = Path(filename).stem
        cleaned_stem = re.sub(r"[-_.]+", " ", stem).strip()
        words = [
            w for w in re.findall(r"\b[A-Za-z]+\b", cleaned_stem)
            if (
                w.lower() not in GENERIC_FILE_WORDS
                and w.lower() not in CLINICIAN_KEYWORDS
                and w.lower() not in CLINICIAN_PREFIXES
                and w.lower() not in STOP_WORDS
            )
        ]
        # Must have at least 2 valid name tokens, each at least 2 characters, first word at least 3 chars
        if len(words) >= 2 and len(words[0]) >= 3 and len(words[1]) >= 2:
            candidate_name = f"{words[0].capitalize()} {words[1].capitalize()}"
            if not _is_clinician_or_facility(candidate_name):
                identity.patient_name = candidate_name
                identity.source = "filename"

    return identity



def validate_document_patient_identity(
    target_patient: Patient,
    raw_text: str,
    filename: str | None = None,
) -> None:
    """
    Validate that an uploaded document belongs to the target patient.
    
    1. Extracts patient identity (name, MRN, DOB) from raw OCR text and filename.
    2. Compares against target patient record.
    3. Raises PatientIdentityMismatchError if there is a clear mismatch.
    4. Passes if identity matches or if patient identity cannot be confidently determined.
    """
    identity = extract_patient_identity(raw_text=raw_text, filename=filename)
    target_full_name = f"{target_patient.first_name} {target_patient.last_name}".strip()

    # Rule 1: Check MRN mismatch
    # If the document explicitly contains an MRN and it does not match target patient's MRN
    if identity.mrn and target_patient.mrn:
        norm_doc_mrn = re.sub(r"[\s\-_]+", "", identity.mrn.upper())
        norm_tgt_mrn = re.sub(r"[\s\-_]+", "", target_patient.mrn.upper())
        if norm_doc_mrn != norm_tgt_mrn:
            logger.warning(
                f"Document patient MRN mismatch: detected '{identity.mrn}' vs target '{target_patient.mrn}' "
                f"for patient {target_patient.patient_id}"
            )
            raise PatientIdentityMismatchError(
                detected_patient_name=identity.patient_name or f"Patient with MRN {identity.mrn}",
                target_patient_name=target_full_name,
                detected_mrn=identity.mrn,
                target_mrn=target_patient.mrn,
                message=(
                    f"Document appears to belong to {identity.patient_name or f'MRN {identity.mrn}'}, "
                    f"but this workspace is for {target_full_name}. The document was not uploaded."
                ),
            )

    # Rule 2: Check Patient Name mismatch
    # If a patient name was identified in the document (from content or filename)
    if identity.patient_name:
        matches = is_name_match(
            detected_name=identity.patient_name,
            target_first=target_patient.first_name,
            target_last=target_patient.last_name,
        )

        if not matches:
            # If name was derived from filename, double-check whether the document content
            # legitimately contains the target patient's name. If content contains target name,
            # content takes precedence.
            if identity.source == "filename" and raw_text:
                if is_name_match(
                    detected_name=raw_text,
                    target_first=target_patient.first_name,
                    target_last=target_patient.last_name,
                ):
                    logger.info("Filename had differing name but content matches target patient; allowing.")
                    return

            logger.warning(
                f"Document patient identity mismatch: detected '{identity.patient_name}' "
                f"vs target '{target_full_name}' for patient {target_patient.patient_id}"
            )
            raise PatientIdentityMismatchError(
                detected_patient_name=identity.patient_name,
                target_patient_name=target_full_name,
                message=(
                    f"Document appears to belong to {identity.patient_name}, "
                    f"but this workspace is for {target_full_name}. The document was not uploaded."
                ),
            )

    # If identity is unavailable or matches target, validation passes.
