"""Focused OCR evaluation with a controlled, apples-to-apples OCR mode.

Manifest CSV columns:
    file_path,ground_truth_path,file_type

For the OCR experiment, this script deliberately bypasses the production
native-PDF/PyMuPDF shortcut and renders every PDF page to an image before
running the selected OCR engine. This makes Tesseract vs PaddleOCR comparable
on identical visual inputs.

Usage:
    python -m scripts.evaluate_ocr <manifest.csv> [--engine tesseract|paddleocr]

The production extract_text() behavior is unchanged.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import fitz

from app.services.ocr_service import (
    _extract_paddle_from_image,
    _extract_paddle_from_pdf,
    _extract_tesseract_from_image,
    _extract_tesseract_from_pdf,
)


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


def _run_controlled_ocr(file_path: Path, file_type: str, engine: str):
    """Run only the selected OCR engine, bypassing native PDF extraction."""
    if file_type == "application/pdf" or file_path.suffix.lower() == ".pdf":
        doc = fitz.open(file_path)
        try:
            if engine == "tesseract":
                return _extract_tesseract_from_pdf(file_path, doc)
            return _extract_paddle_from_pdf(doc)
        finally:
            doc.close()

    if (
        file_type in ("image/jpeg", "image/png", "image/jpg")
        or file_path.suffix.lower() in (".jpg", ".jpeg", ".png")
    ):
        if engine == "tesseract":
            return _extract_tesseract_from_image(file_path)
        return _extract_paddle_from_image(file_path)

    raise ValueError(f"Unsupported file type for OCR: {file_type}")


def main(manifest_path: str, engine: str) -> None:
    rows = list(csv.DictReader(Path(manifest_path).open(encoding="utf-8")))
    if not rows:
        raise SystemExit("Manifest is empty.")

    cer_values: list[float] = []
    wer_values: list[float] = []
    confidence_values: list[float] = []

    print(f"Controlled OCR engine: {engine}")
    print("Native PDF extraction is bypassed for this evaluation.\n")

    for row in rows:
        result = _run_controlled_ocr(
            Path(row["file_path"]),
            row["file_type"],
            engine,
        )
        reference = Path(row["ground_truth_path"]).read_text(encoding="utf-8")
        cer = _char_error_rate(reference, result.raw_text)
        wer = _word_error_rate(reference, result.raw_text)
        cer_values.append(cer)
        wer_values.append(wer)
        if result.confidence is not None:
            confidence_values.append(result.confidence)

        print(
            f'{row["file_path"]}: CER={cer:.4f} WER={wer:.4f} '
            f'confidence={result.confidence} engine={result.ocr_engine}'
        )

    print("\nAggregate")
    print(f"engine={engine}")
    print(f"documents={len(rows)}")
    print(f"mean_CER={sum(cer_values) / len(cer_values):.4f}")
    print(f"mean_WER={sum(wer_values) / len(wer_values):.4f}")
    if confidence_values:
        print(
            "mean_OCR_confidence="
            f"{sum(confidence_values) / len(confidence_values):.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="Path to the OCR evaluation manifest CSV")
    parser.add_argument(
        "--engine",
        choices=("tesseract", "paddleocr"),
        default="tesseract",
        help="OCR engine to evaluate (default: tesseract)",
    )
    args = parser.parse_args()
    main(args.manifest, args.engine)
