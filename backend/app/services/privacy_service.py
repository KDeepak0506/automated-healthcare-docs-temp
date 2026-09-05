import re
import logging
from dataclasses import dataclass, field

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

logger = logging.getLogger(__name__)


@dataclass
class SanitizationResult:
    sanitized_text: str
    entities_detected: int
    entity_counts: dict[str, int] = field(default_factory=dict)


class PrivacyService:
    def __init__(self) -> None:
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        self.target_entities = [
            "PATIENT_NAME",
            "MRN",
            "UHID",
            "PATIENT_ID",
            "INSURANCE_ID",
            "POLICY_NUMBER",
            "PHONE_NUMBER",
            "EMAIL",
            "ADDRESS",
            "DATE_OF_BIRTH",
            "IP_NUMBER",
            "ACCOUNT_NUMBER",
        ]

        self._register_custom_recognizers()

    def _register_custom_recognizers(self) -> None:
        # Contextual header pattern recognizers for high precision
        recognizers = [
            PatternRecognizer(
                supported_entity="PATIENT_NAME",
                patterns=[
                    Pattern("patient_name_prefix", r"(?i)(?:Patient\s*Name|Patient|Name)\s*[:#]\s*([A-Za-z\s.'-]+)", 0.85),
                ],
            ),
            PatternRecognizer(
                supported_entity="MRN",
                patterns=[
                    Pattern("mrn_prefix", r"(?i)(?:MRN|Medical\s*Record\s*(?:No|Number|#)?)\s*[:#]?\s*([A-Za-z0-9-]+)", 0.9),
                ],
            ),
            PatternRecognizer(
                supported_entity="UHID",
                patterns=[
                    Pattern("uhid_prefix", r"(?i)(?:UHID|Unique\s*Health\s*ID)\s*[:#]?\s*([A-Za-z0-9-]+)", 0.9),
                ],
            ),
            PatternRecognizer(
                supported_entity="PATIENT_ID",
                patterns=[
                    Pattern("patient_id_prefix", r"(?i)(?:Patient\s*ID|Pat\s*ID)\s*[:#]?\s*([A-Za-z0-9-]+)", 0.9),
                ],
            ),
            PatternRecognizer(
                supported_entity="INSURANCE_ID",
                patterns=[
                    Pattern("insurance_id_prefix", r"(?i)(?:Insurance\s*(?:ID|No|Number|#)?)\s*[:#]?\s*([A-Za-z0-9-]+)", 0.85),
                ],
            ),
            PatternRecognizer(
                supported_entity="POLICY_NUMBER",
                patterns=[
                    Pattern("policy_num_prefix", r"(?i)(?:Policy\s*(?:No|Number|#)?)\s*[:#]?\s*([A-Za-z0-9-]+)", 0.85),
                ],
            ),
            PatternRecognizer(
                supported_entity="PHONE_NUMBER",
                patterns=[
                    Pattern("phone_prefix", r"(?i)(?:Phone|Mobile|Tel|Cell)\s*[:#]?\s*(\+?\d[\d\s-]{7,15}\d)", 0.85),
                ],
            ),
            PatternRecognizer(
                supported_entity="EMAIL",
                patterns=[
                    Pattern("email_prefix", r"(?i)(?:Email|E-mail)\s*[:#]?\s*([\w.-]+@[\w.-]+\.\w+)", 0.9),
                ],
            ),
            PatternRecognizer(
                supported_entity="ADDRESS",
                patterns=[
                    Pattern("address_prefix", r"(?i)(?:Address|Residing\s*at)\s*[:#]?\s*([^\n]+)", 0.8),
                ],
            ),
            PatternRecognizer(
                supported_entity="DATE_OF_BIRTH",
                patterns=[
                    Pattern("dob_prefix", r"(?i)(?:DOB|Date\s*of\s*Birth|Birth\s*Date)\s*[:#]?\s*(\d{1,4}[/-]\d{1,2}[/-]\d{1,4})", 0.95),
                ],
            ),
            PatternRecognizer(
                supported_entity="IP_NUMBER",
                patterns=[
                    Pattern("ip_num_prefix", r"(?i)(?:IP\s*(?:No|Number|#)?|Inpatient\s*(?:No|Number|#)?)\s*[:#]?\s*([A-Za-z0-9-]+)", 0.85),
                ],
            ),
            PatternRecognizer(
                supported_entity="ACCOUNT_NUMBER",
                patterns=[
                    Pattern("acc_num_prefix", r"(?i)(?:Account\s*(?:No|Number|#)?|Acc\s*(?:No|Number|#)?)\s*[:#]?\s*([A-Za-z0-9-]+)", 0.85),
                ],
            ),
        ]

        for rec in recognizers:
            self.analyzer.registry.add_recognizer(rec)

    def sanitize(self, text: str) -> SanitizationResult:
        if not text or not text.strip():
            return SanitizationResult(
                sanitized_text=text or "",
                entities_detected=0,
                entity_counts={},
            )

        try:
            # We also include standard entities from presidio like PERSON, PHONE_NUMBER, EMAIL_ADDRESS, LOCATION, DATE_TIME
            entities_to_analyze = self.target_entities + [
                "PERSON",
                "PHONE_NUMBER",
                "EMAIL_ADDRESS",
                "LOCATION",
                "DATE_TIME",
                "US_SSN",
            ]

            analyzer_results = self.analyzer.analyze(
                text=text,
                language="en",
                entities=entities_to_analyze,
                score_threshold=0.4,
            )

            # Filter out false positives for medical lab values / diagnoses
            filtered_results = []
            for res in analyzer_results:
                matched_text = text[res.start:res.end].strip()
                # Exclude medical numbers/lab values like 8.2%, 1.4 mg/dL, 500 mg, Type 2 Diabetes, Metformin
                if re.search(r"\b(?:\d+\.?\d*\s*(?:%|mg/dL|g/dL|mmol/L|mcg|mg|mL|IU/L|uIU/mL)|Type\s*\d|Metformin|Glucose|HbA1c|Creatinine)\b", matched_text, re.IGNORECASE):
                    continue
                filtered_results.append(res)

            # Define operators for deterministic tag redaction
            operators = {
                "PATIENT_NAME": OperatorConfig("replace", {"new_value": "[PATIENT]"}),
                "PERSON": OperatorConfig("replace", {"new_value": "[PATIENT]"}),
                "MRN": OperatorConfig("replace", {"new_value": "[MRN]"}),
                "UHID": OperatorConfig("replace", {"new_value": "[UHID]"}),
                "PATIENT_ID": OperatorConfig("replace", {"new_value": "[PATIENT_ID]"}),
                "INSURANCE_ID": OperatorConfig("replace", {"new_value": "[INSURANCE_ID]"}),
                "POLICY_NUMBER": OperatorConfig("replace", {"new_value": "[POLICY_NUMBER]"}),
                "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
                "EMAIL": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
                "ADDRESS": OperatorConfig("replace", {"new_value": "[ADDRESS]"}),
                "LOCATION": OperatorConfig("replace", {"new_value": "[ADDRESS]"}),
                "DATE_OF_BIRTH": OperatorConfig("replace", {"new_value": "[DATE]"}),
                "DATE_TIME": OperatorConfig("replace", {"new_value": "[DATE]"}),
                "IP_NUMBER": OperatorConfig("replace", {"new_value": "[IP_NUMBER]"}),
                "ACCOUNT_NUMBER": OperatorConfig("replace", {"new_value": "[ACCOUNT_NUMBER]"}),
                "US_SSN": OperatorConfig("replace", {"new_value": "[SSN]"}),
            }

            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=filtered_results,
                operators=operators,
            )

            # Perform high-precision Regex replacements for contextual key-value headers to ensure complete sanitization
            sanitized_text = anonymized_result.text

            # Header regex sanitizations
            replacements = [
                (r"(?i)(Patient\s*Name\s*[:#]\s*)[^\n]+", r"\1[PATIENT]"),
                (r"(?i)(DOB\s*[:#]\s*)[^\n]+", r"\1[DATE]"),
                (r"(?i)(Date\s*of\s*Birth\s*[:#]\s*)[^\n]+", r"\1[DATE]"),
                (r"(?i)(MRN\s*[:#]\s*)[^\n]+", r"\1[MRN]"),
                (r"(?i)(UHID\s*[:#]\s*)[^\n]+", r"\1[UHID]"),
                (r"(?i)(Patient\s*ID\s*[:#]\s*)[^\n]+", r"\1[PATIENT_ID]"),
                (r"(?i)(Insurance\s*(?:ID|No|Number)?\s*[:#]\s*)[^\n]+", r"\1[INSURANCE_ID]"),
                (r"(?i)(Policy\s*(?:No|Number)?\s*[:#]\s*)[^\n]+", r"\1[POLICY_NUMBER]"),
                (r"(?i)(Phone\s*[:#]\s*)[^\n]+", r"\1[PHONE]"),
                (r"(?i)(Email\s*[:#]\s*)[^\n]+", r"\1[EMAIL]"),
                (r"(?i)(Address\s*[:#]\s*)[^\n]+", r"\1[ADDRESS]"),
                (r"(?i)(IP\s*(?:No|Number)?\s*[:#]\s*)[^\n]+", r"\1[IP_NUMBER]"),
                (r"(?i)(Account\s*(?:No|Number)?\s*[:#]\s*)[^\n]+", r"\1[ACCOUNT_NUMBER]"),
                (r"\b\d{3}-\d{2}-\d{4}\b", r"[SSN]"),
            ]

            for pattern, repl in replacements:
                sanitized_text = re.sub(pattern, repl, sanitized_text)

            # Calculate entity counts safely
            counts: dict[str, int] = {}
            for res in filtered_results:
                entity_type = res.entity_type
                counts[entity_type] = counts.get(entity_type, 0) + 1

            total_entities = len(filtered_results)

            return SanitizationResult(
                sanitized_text=sanitized_text,
                entities_detected=total_entities,
                entity_counts=counts,
            )
        except Exception as exc:
            logger.error(f"Privacy processing failed during sanitization: {exc}")
            raise


# Singleton instance for service usage
privacy_service = PrivacyService()
