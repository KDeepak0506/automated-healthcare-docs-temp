import logging
import re
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.document_entity import DocumentEntity

logger = logging.getLogger(__name__)

CLINICAL_LABELS = [
    "disease",
    "medication",
    "dosage",
    "symptom",
    "procedure",
    "anatomical structure",
    "lab test",
    "lab value",
]

GENERIC_HEADERS = {
    "physical examination",
    "biochemical examination",
    "microscopic examination",
    "biochemical & laboratory examination",
    "physical examination & vital signs",
    "procedures & investigations",
    "clinical history & diagnosis",
    "patient information",
    "clinical laboratory report",
    "laboratory report",
    "specimen details",
    "general findings",
    "notes & interpretation",
}

LAB_VALUE_PATTERNS = [
    # Blood pressure e.g. 140/90 mmHg or 140/90
    (r"\b\d{2,3}/\d{2,3}\s*(?:mmHg|mm Hg)?\b", "lab value", 0.95),
    # Numbers with lab units e.g. 192 mg/dL, 8.2 %, 1.41 mg/dL, 139 mmol/L, 57.10 mL/min, 9.5 mg/dL, 21 U/L, 98.6 °F, 78 bpm, 2-4 / hpf
    (r"\b\d+(?:\.\d+)?\s*(?:%|mg/dL|g/dL|mcg/dL|mmol/L|umol/L|µmol/L|mEq/L|U/L|IU/L|uIU/mL|mIU/mL|ng/mL|pg/mL|bpm|°F|°C|mmHg|mm Hg|g%|vol%|cells/uL|/uL|/hpf|/HPF|mL/min|mL/min/1\.73m2)\b", "lab value", 0.95),
    # Cell count / Scientific notation / counts with commas e.g. 6.8 x 10^3 / uL, 240,000 / uL
    (r"\b\d+(?:\.\d+)?\s*(?:x\s*10\^(?:3|6|9)|x\s*10\*(?:3|6|9)|,\d{3})\s*(?:/uL|/L|/mL|/hpf)?\b", "lab value", 0.93),
]

KNOWN_LAB_TEST_KEYWORDS = [
    "Glucose", "Fasting Blood Sugar", "HbA1c", "HbA1C", "Glycated Hemoglobin",
    "Serum Creatinine", "Creatinine", "Blood Urea Nitrogen", "BUN", "eGFR",
    "Total Cholesterol", "HDL Cholesterol", "LDL Cholesterol", "Triglycerides",
    "Serum Sodium", "Sodium", "Serum Potassium", "Potassium", "Serum Chloride", "Chloride", "Serum Calcium", "Calcium",
    "Hemoglobin", "Hb", "WBC Count", "WBC", "Platelet Count", "Platelets",
    "Blood Pressure", "Heart Rate", "Temperature", "Bilirubin", "ALT", "AST", "ALP",
    "TSH", "T3", "T4", "Uric Acid", "Microscopy", "Pus cells", "ESR", "CRP",
]

MEDICATION_KEYWORDS = [
    "Metformin", "Aspirin", "Lisinopril", "Amoxicillin", "Ibuprofen",
    "Paracetamol", "Atorvastatin", "Omeprazole", "Insulin", "Levothyroxine",
    "Amlodipine", "Metoprolol", "Losartan", "Gabapentin", "Hydrochlorothiazide"
]

FREQUENCY_PATTERNS = r"(?i)\b(?:daily|twice\s+daily|once\s+daily|three\s+times\s+daily|BID|TID|QID|QD|PRN|every\s+\d+\s+hours|morning|evening|night|p\.o\.|po|orally)\b"


class ClinicalNERService:
    """
    Service for Clinical Information Extraction / Biomedical NER (M4).
    Uses GLiNER (GLiNER-Small/BioMed) combined with deterministic medical & lab extraction rules.
    STRICTLY consumes sanitized_text ONLY (never raw OCR / raw PHI).
    """

    def __init__(self, model_name: str = "urchade/gliner_small-v2.1"):
        self.model_name = model_name
        self._model = None
        self._model_loaded = False

    def _get_model(self) -> Any:
        if not self._model_loaded:
            self._model_loaded = True
            try:
                from gliner import GLiNER
                logger.info(f"Loading GLiNER model: {self.model_name}")
                self._model = GLiNER.from_pretrained(self.model_name)
                logger.info("GLiNER model loaded successfully")
            except Exception as e:
                logger.warning(
                    f"Failed to load GLiNER model {self.model_name}: {e}. "
                    "Will use pattern-based clinical extraction fallback."
                )
                self._model = None
        return self._model

    def preprocess_text(self, sanitized_text: str) -> str:
        """
        Lightweight normalization to assist NER without altering character offsets unnecessarily.
        """
        if not sanitized_text:
            return ""
        return sanitized_text

    def extract_gliner_entities(self, text: str) -> list[dict]:
        """
        Extract clinical entities from text using GLiNER model.
        Returns a list of dicts with float confidence scores.
        """
        model = self._get_model()
        if model is None:
            return []
        try:
            entities = model.predict_entities(text, CLINICAL_LABELS, threshold=0.30)
            results = []
            for ent in entities:
                score = float(ent.get("score", 0.0))
                results.append({
                    "text": ent["text"],
                    "label": ent["label"],
                    "confidence": score,
                    "start_char": ent.get("start"),
                    "end_char": ent.get("end"),
                })
            return results
        except Exception as exc:
            logger.error(f"GLiNER inference error: {exc}")
            return []

    def extract_lab_values_and_rules(self, text: str) -> tuple[list[dict], list[str]]:
        """
        Deterministic rule-based extraction for laboratory values, measurements, contextual dosages, and lab tests.
        """
        results = []
        lab_value_texts = []
        if not text:
            return results, lab_value_texts

        # 1. Match numeric lab values & measurements with units
        for pattern, label, conf in LAB_VALUE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                val_text = match.group(0).strip()
                lab_value_texts.append(val_text)
                results.append({
                    "text": val_text,
                    "label": label,
                    "confidence": float(conf),
                    "start_char": match.start(),
                    "end_char": match.end(),
                })

        # 2. Match contextual dosages (ONLY when medication or frequency context is present)
        dosage_pattern = r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|tablets?|capsules?)\b"
        for match in re.finditer(dosage_pattern, text, re.IGNORECASE):
            dos_text = match.group(0).strip()
            start = match.start()
            end = match.end()

            # Skip if followed immediately by '/' (e.g. mg/dL, g/dL, mL/min)
            if end < len(text) and text[end] == '/':
                continue

            # Check if substring of any lab_value
            is_lab_sub = False
            for lv in lab_value_texts:
                if dos_text in lv and ("/" in lv or "%" in lv or "mEq" in lv or "bpm" in lv or "°" in lv or "mmHg" in lv):
                    is_lab_sub = True
                    break
            if is_lab_sub:
                continue

            # Context check: medication before OR administration frequency after
            context_before = text[max(0, start - 40):start]
            context_after = text[end:min(len(text), end + 40)]
            has_med = any(re.search(r"\b" + re.escape(med) + r"\b", context_before, re.IGNORECASE) for med in MEDICATION_KEYWORDS)
            has_freq = bool(re.search(FREQUENCY_PATTERNS, context_after))

            if has_med or has_freq or "prescribe" in context_before.lower() or "take" in context_before.lower() or "dose" in context_before.lower():
                results.append({
                    "text": dos_text,
                    "label": "dosage",
                    "confidence": 0.90,
                    "start_char": start,
                    "end_char": end,
                })

        # 3. Match known lab test keywords in text
        for kw in KNOWN_LAB_TEST_KEYWORDS:
            for match in re.finditer(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE):
                kw_text = match.group(0).strip()
                lbl = "lab test" if not any(v in kw_text.lower() for v in ["pressure", "rate", "temperature"]) else "procedure"
                results.append({
                    "text": kw_text,
                    "label": lbl,
                    "confidence": 0.92,
                    "start_char": match.start(),
                    "end_char": match.end(),
                })

        return results, lab_value_texts

    def clean_lab_test_span(self, text: str) -> str:
        """
        Clean long GLiNER lab test spans (e.g., 'Decreased ESR is seen in') to concise test names (e.g. 'ESR').
        """
        text = text.strip(" \t\n\r:,-=.")
        if not text:
            return ""

        if len(text) > 20 or any(w in text.lower() for w in ["seen", "interfere", "influence", "factors", "measured", "result", "interpretation", "notes", "clinical"]):
            for kw in KNOWN_LAB_TEST_KEYWORDS:
                if re.search(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE):
                    return kw
            if len(text) > 35 or re.search(r"\b(?:is|are|was|were|has|have|seen|with|for|that|the|of)\b", text, re.IGNORECASE):
                return ""
        return text

    def _fallback_extract(self, text: str) -> list[dict]:
        """
        Rule/pattern fallback when GLiNER model is not loaded or fails.
        """
        patterns = [
            (r"\b(hypertension|diabetes|type 2 diabetes|asthma|pneumonia|arrhythmia|ischemia|headache|fever|cough|chest pain)\b", "disease", 0.88),
            (r"\b(aspirin|metformin|lisinopril|amoxicillin|ibuprofen|paracetamol|atorvastatin|omeprazole)\b", "medication", 0.92),
            (r"\b(\d+\s*(mg|g|mcg|ml|tablets?|capsules?))\b", "dosage", 0.85),
            (r"\b(shortness of breath|fatigue|dizziness|nausea|pain|swelling|rash)\b", "symptom", 0.84),
            (r"\b(ecg|electrocardiogram|chest x-ray|mri|ct scan|biopsy|endoscopy|blood test|angioplasty)\b", "procedure", 0.89),
            (r"\b(heart|lungs?|liver|kidney|brain|abdomen|chest|left arm)\b", "anatomical structure", 0.82),
            (r"\b(hemoglobin|hba1c|white blood cell count|serum creatinine|glucose|cholesterol|wbc count|platelet count)\b", "lab test", 0.87),
        ]

        results = []
        for pattern, label, conf in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                results.append({
                    "text": match.group(0),
                    "label": label,
                    "confidence": float(conf),
                    "start_char": match.start(),
                    "end_char": match.end(),
                })
        return results

    def normalize_entities(self, raw_entities: list[dict], lab_value_texts: list[str]) -> list[dict]:
        """
        Clean entity text, normalize long GLiNER spans, and filter generic heading false positives.
        """
        cleaned = []
        for ent in raw_entities:
            label = ent["label"]
            text = ent["text"].strip(" \t\n\r:,-=.")

            if not text:
                continue

            # Remove generic section headers misclassified as procedure or lab test
            if text.lower() in GENERIC_HEADERS:
                continue

            # Clean long lab test spans from GLiNER
            if label == "lab test":
                text = self.clean_lab_test_span(text)
                if not text:
                    continue

            # Prevent lab value texts from being misclassified as dosage
            if label == "dosage":
                if text in lab_value_texts or any(text in lv and ("/" in lv or "%" in lv or "mEq" in lv or "bpm" in lv or "°" in lv or "mmHg" in lv) for lv in lab_value_texts):
                    continue

            # Prevent low-confidence GLiNER lab value misclassifications for test keywords
            if label == "lab value" and text in KNOWN_LAB_TEST_KEYWORDS and ent["confidence"] < 0.50:
                continue

            ent_copy = dict(ent)
            ent_copy["text"] = text
            ent_copy["confidence"] = float(ent_copy["confidence"])
            cleaned.append(ent_copy)
        return cleaned

    def deduplicate_entities(self, entities: list[dict]) -> list[dict]:
        """
        Deduplicate identical entities or redundant overlapping spans for the same label.
        Preserves distinct entity labels (e.g. 'HbA1c' as lab test vs '8.2 %' as lab value).
        """
        if not entities:
            return []

        # Prefer higher confidence entities
        sorted_ents = sorted(entities, key=lambda x: (x.get("start_char") or 0, -x["confidence"], -len(x["text"])))
        deduped = []
        seen_keys = set()

        for ent in sorted_ents:
            key = (ent["text"].lower(), ent["label"])
            if key in seen_keys:
                continue

            # Check for substring overlap with existing entity of SAME label
            is_sub = False
            for existing in deduped:
                if existing["label"] == ent["label"]:
                    if ent["text"].lower() in existing["text"].lower() and ent["text"].lower() != existing["text"].lower():
                        is_sub = True
                        break
            if is_sub:
                continue

            seen_keys.add(key)
            deduped.append(ent)

        return deduped

    def extract_entities(self, sanitized_text: str) -> list[dict]:
        """
        Extract clinical entities strictly from sanitized_text ONLY.
        """
        if not sanitized_text or not sanitized_text.strip():
            return []

        preprocessed = self.preprocess_text(sanitized_text)
        gliner_ents = self.extract_gliner_entities(preprocessed)

        # Use fallback pattern extraction if GLiNER model is not loaded
        if not gliner_ents and self._get_model() is None:
            gliner_ents = self._fallback_extract(preprocessed)

        rule_ents, lab_value_texts = self.extract_lab_values_and_rules(preprocessed)

        combined = gliner_ents + rule_ents
        normalized = self.normalize_entities(combined, lab_value_texts)
        final_ents = self.deduplicate_entities(normalized)
        return final_ents

    def extract_and_store_entities(
        self,
        db: Session,
        document_id: UUID,
        sanitized_text: str,
    ) -> list[DocumentEntity]:
        """
        Extract clinical entities from sanitized_text and persist them in document_entities.
        """
        # Delete existing entities for idempotent reprocessing
        db.query(DocumentEntity).filter(DocumentEntity.document_id == document_id).delete()

        extracted = self.extract_entities(sanitized_text)
        entities = []
        for item in extracted:
            entity = DocumentEntity(
                document_id=document_id,
                text=item["text"],
                label=item["label"],
                confidence=float(item["confidence"]),
                start_char=item.get("start_char"),
                end_char=item.get("end_char"),
            )
            db.add(entity)
            entities.append(entity)

        db.commit()
        for e in entities:
            db.refresh(e)

        logger.info(f"Persisted {len(entities)} clinical entities for document {document_id}")
        return entities


clinical_ner_service = ClinicalNERService()
