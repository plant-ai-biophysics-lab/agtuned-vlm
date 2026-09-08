"""Ablation stage, "oracle-leakage" step (Q5) — quantifies how much of the
description-augmented "oracle upper bound" condition (src/desc_augmented_
vision_v2.py) is built from descriptions that caption real images of the
correct class, rather than from class-agnostic web/encyclopedia text.

dataset/agml_classification.jsonl's metadata.grounding_status records how
each class's synthesized_description was built (src/data_generation/
generate.py):
  grounded_web_and_vision — merged from web text AND captions of real
                             images of this class (leak risk)
  grounded_vision_only    — captions of real images of this class only,
                             no web text at all (leak risk)
  grounded_web_only       — web/encyclopedia text only, no image captioning
  no_source_found         — empty description, generic fallback used
  healthy_skip            — healthy/no-stress class, nothing to describe

The first two statuses mean the "oracle" description can contain visually
distinguishing detail the model would otherwise have to notice unaided —
privileged information tied to the specific images, not general domain
knowledge. This module tabulates how many evaluated classes fall into each
bucket and breaks out desc-aug F1 by bucket, so the paper can report
whether "exceeding the oracle upper bound" concentrates in the leaky
subset.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

LEAKY_STATUSES = {"grounded_vision_only", "grounded_web_and_vision"}


def load_grounding_status_map(classification_jsonl: Path) -> dict[tuple[str, str], str]:
    """{(agml_dataset, class_label): grounding_status}"""
    out = {}
    with open(classification_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("metadata", {})
            key = (meta.get("agml_dataset", ""), meta.get("class_label", ""))
            out[key] = meta.get("grounding_status", "unknown")
    return out


def tally_grounding_status(status_map: dict[tuple[str, str], str]) -> list[dict]:
    """Class-level tally (one row per (agml_dataset, class_label) — the
    grounding status is per-class, not per-image, so "fraction of evaluated
    samples" here really means fraction of evaluated CLASSES, weighted
    equally regardless of how many images/class were sampled)."""
    counts = defaultdict(int)
    for status in status_map.values():
        counts[status] += 1
    total = len(status_map)
    return sorted(
        [{"grounding_status": s, "n_classes": n, "fraction": n / total if total else 0.0,
          "leaky": s in LEAKY_STATUSES} for s, n in counts.items()],
        key=lambda r: -r["n_classes"],
    )


def join_desc_aug_rows_with_grounding(
    desc_aug_ckpt_csv: Path, status_map: dict[tuple[str, str], str],
) -> list[dict]:
    """Every row of a desc_augmented_vision(_v2).py checkpoint CSV, plus a
    `grounding_status`/`leaky` column joined on (agml_dataset, class_label) —
    same key desc_augmented_vision_v2.py::load_description_map already uses,
    confirmed identical."""
    out = []
    with open(desc_aug_ckpt_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["agml_dataset"], row["class_label"])
            status = status_map.get(key, "unknown")
            row = dict(row)
            row["grounding_status"] = status
            row["leaky"] = status in LEAKY_STATUSES
            out.append(row)
    return out


def f1_by_grounding_status(rows: list[dict]) -> dict[str, dict]:
    """{grounding_status: {n, accuracy, judged_accuracy}} — direct-match
    accuracy (`correct`) and judged accuracy (`judged_correct`, falling back
    to `correct` if the judge phase wasn't run), broken out by grounding
    status, so a per-status gap is visible without re-running the judge."""
    by_status = defaultdict(list)
    for r in rows:
        by_status[r["grounding_status"]].append(r)
    out = {}
    for status, grp in by_status.items():
        correct = [int(r.get("correct", 0) or 0) for r in grp]
        judged = [
            int(r["judged_correct"]) if r.get("judged_correct", "") not in ("", None)
            else int(r.get("correct", 0) or 0)
            for r in grp
        ]
        out[status] = {
            "n": len(grp),
            "accuracy": sum(correct) / len(correct) if correct else None,
            "judged_accuracy": sum(judged) / len(judged) if judged else None,
        }
    return out


def rows_by_condition_for_chart(rows: list[dict]) -> dict[str, list[dict]]:
    """Buckets desc-aug rows into "Leaky (vision-derived)" vs. "Non-leaky
    (web-only/none)" conditions, matching the {condition: [row, ...]} shape
    src.common.method_comparison's plotting functions expect (they read
    `judged_correct` per the standard CKPT_FIELDS row shape, which these
    rows already carry unmodified from the checkpoint)."""
    out: dict[str, list[dict]] = {"Leaky (vision-derived)": [], "Non-leaky (web-only/none)": []}
    for r in rows:
        out["Leaky (vision-derived)" if r["leaky"] else "Non-leaky (web-only/none)"].append(r)
    return out
