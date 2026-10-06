"""Focused OCR evaluation: CER/WER on a manifest of scanned documents.

Manifest CSV columns:
    file_path,ground_truth_path,file_type

The same manifest can be evaluated with OCR_ENGINE=tesseract and
OCR_ENGINE=paddleocr. Keep the ground-truth files human-verified.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from app.services.ocr_service import extract_text


def _edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (token_a != token_b),
                )
            )
        previous = current
    return previous[-1]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _char_error_rate(reference: str, hypothesis: str) -> float:
    reference = _normalize(reference)
    hypothesis = _normalize(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(list(reference), list(hypothesis)) / len(reference)


def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = _normalize(reference).split()
    hyp_words = _normalize(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def main(manifest_path: str) -> None:
    rows = list(csv.DictReader(Path(manifest_path).open(encoding="utf-8")))
    if not rows:
        raise SystemExit("Manifest is empty.")

    cer_values: list[float] = []
    wer_values: list[float] = []
    confidence_values: list[float] = []

    for row in rows:
        result = extract_text(Path(row["file_path"]), row["file_type"])
        reference = Path(row["ground_truth_path"]).read_text(encoding="utf-8")
        cer = _char_error_rate(reference, result.raw_text)
        wer = _word_error_rate(reference, result.raw_text)
        cer_values.append(cer)
        wer_values.append(wer)
        if result.confidence is not None:
            confidence_values.append(result.confidence)

        print(
            f'{row["file_path"]}: CER={cer:.4f} WER={wer:.4f} '
            f'confidence={result.confidence}'
        )

    print("\nAggregate")
    print(f"documents={len(rows)}")
    print(f"mean_CER={sum(cer_values) / len(cer_values):.4f}")
    print(f"mean_WER={sum(wer_values) / len(wer_values):.4f}")
    if confidence_values:
        print(
            f"mean_OCR_confidence="
            f"{sum(confidence_values) / len(confidence_values):.4f}"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m scripts.evaluate_ocr <manifest.csv>")
    main(sys.argv[1])
