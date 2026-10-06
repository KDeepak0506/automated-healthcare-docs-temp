"""Focused clinical NER evaluation using exact span + label matching.

JSONL format, one document per line:
{"text":"...", "entities":[
  {"text":"diabetes","label":"disease","start":0,"end":8}
]}

Entity spans should use Python string offsets from the exact text field.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from app.services.ner_service import ClinicalNERService


def _evaluate(gold: list[dict], predicted: list[dict]) -> tuple[int, int, int]:
    gold_set = {
        (e["start"], e["end"], e["label"].strip().lower())
        for e in gold
    }
    pred_set = {
        (e.get("start_char"), e.get("end_char"), e["label"].strip().lower())
        for e in predicted
    }
    tp = len(gold_set & pred_set)
    return tp, len(pred_set) - tp, len(gold_set) - tp


def main(dataset_path: str) -> None:
    service = ClinicalNERService()
    totals = Counter()
    per_label = Counter()

    for line in Path(dataset_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        predicted = service.extract_gliner_entities(item["text"])
        tp, fp, fn = _evaluate(item["entities"], predicted)
        totals.update(tp=tp, fp=fp, fn=fn)

        gold_labels = {e["label"].strip().lower() for e in item["entities"]}
        for label in gold_labels:
            label_gold = [
                e for e in item["entities"]
                if e["label"].strip().lower() == label
            ]
            label_pred = [
                e for e in predicted
                if e["label"].strip().lower() == label
            ]
            ltp, lfp, lfn = _evaluate(label_gold, label_pred)
            per_label[label, "tp"] += ltp
            per_label[label, "fp"] += lfp
            per_label[label, "fn"] += lfn

    def scores(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return precision, recall, f1

    p, r, f1 = scores(totals["tp"], totals["fp"], totals["fn"])
    print(f"micro_precision={p:.4f}")
    print(f"micro_recall={r:.4f}")
    print(f"micro_f1={f1:.4f}")

    print("\nPer label")
    for label in sorted({key[0] for key in per_label}):
        p, r, f1 = scores(
            per_label[label, "tp"],
            per_label[label, "fp"],
            per_label[label, "fn"],
        )
        print(f"{label}: precision={p:.4f} recall={r:.4f} f1={f1:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m scripts.evaluate_ner <dataset.jsonl>")
    main(sys.argv[1])
