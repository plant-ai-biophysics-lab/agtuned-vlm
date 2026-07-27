"""
Full, untruncated conversation transcripts for a handful of example images —
cot_yaml, self_cot, and desc_aug side by side — for manual PI review.

Unlike the response-card PNGs (which truncate long text to fit a fixed box),
this reconstructs the exact prompts sent to the model and copies predictions
verbatim into a plain-text Markdown document with no length limit.

Sources (must already exist):
  outputs/cot_step_analysis/swap_order/ckpt_swap_judged.csv   cot_yaml (rotation 0)
  outputs/cot_vision/cot_cls_<model>.csv                      self_cot final answers
  outputs/desc_augmented_vision/aug_cls_all_<model>.csv       desc_aug final answers

Known gap: self_cot's stage-1 description was never persisted to disk (only
the final classification prediction was saved) — see caveats in
outputs/verifier_analysis/verifier_findings.pdf. That turn is marked
"[not persisted]" below rather than invented.

Usage:
  uv run python src/verifier_analysis_transcripts.py --model google/gemma-4-26B-A4B-it
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
import yaml
from pathlib import Path

csv.field_size_limit(sys.maxsize)

PROMPTS = yaml.safe_load(open(Path(__file__).parent / "config" / "prompts.yaml"))


def img_key(dataset: str, image_name: str, class_label: str) -> tuple:
    return (dataset, os.path.basename(image_name), class_label)


def cls_question(task_type: str, crop: str | None) -> str:
    if task_type == "weed_identification":
        inner = PROMPTS["visual_classification_weed"]
    else:
        inner = PROMPTS["visual_classification_disease"].format(crop=crop or "plant")
    return PROMPTS["assessment_classification"].format(question=inner)


def desc_question(task_type: str, crop: str | None) -> str:
    if task_type == "weed_identification":
        return PROMPTS["visual_description_weed"]
    q = PROMPTS["visual_description_disease"]
    return q.format(crop=crop or "plant") if "{crop}" in q else q


def desc_aug_question(task_type: str, crop: str | None, description: str) -> str:
    if task_type == "weed_identification":
        inner = PROMPTS["visual_classification_weed"]
    else:
        inner = PROMPTS["visual_classification_disease"].format(crop=crop or "plant")
    return PROMPTS["desc_augmented_classification"].format(description=description, question=inner)


def split_step_qa(reasoning: str) -> list[tuple[str, str]]:
    """Parse a cumulative reasoning blob (cot_step_analysis's "Q: ...\\nA: ..."
    format, steps joined by blank lines) into an ordered list of (question,
    answer) pairs, one per CoT step. Mirrors verifier_analysis.extract_step_qa
    but returns every step instead of only the last."""
    blocks = re.split(r"\n\nQ:\s*", reasoning.strip())
    pairs = []
    for i, block in enumerate(blocks):
        text = block.strip()
        if i == 0:
            if text.startswith("Q: "):
                text = text[3:]
            elif text.startswith("Q:"):
                text = text[2:]
        # blocks after the first already have "Q:" consumed by the split regex
        if "\nA: " in text:
            q, a = text.split("\nA: ", 1)
        elif "\nA:" in text:
            q, a = text.split("\nA:", 1)
        else:
            q, a = text, ""
        pairs.append((q.strip(), a.strip()))
    return pairs


def load_swap(swap_path: Path, rotation: str = "0") -> dict[tuple, dict]:
    by_key: dict[tuple, dict] = {}
    for r in csv.DictReader(open(swap_path, encoding="utf-8")):
        if r.get("mode") != "cumulative" or r.get("rotation") != rotation:
            continue
        if r.get("judged_correct", "") == "":
            continue
        k = img_key(r["agml_dataset"], r["image_name"], r["class_label"])
        step = int(r["step_idx"])
        prev = by_key.get(k)
        if prev is None or step > int(prev["step_idx"]):
            by_key[k] = r
    return by_key


def load_self_cot(path: Path) -> dict[tuple, dict]:
    return {
        img_key(r["agml_dataset"], r["image_name"], r["class_label"]): r
        for r in csv.DictReader(open(path, encoding="utf-8"))
        if r.get("condition") == "self_cot"
    }


def load_desc_aug(path: Path) -> dict[tuple, dict]:
    return {
        img_key(r["agml_dataset"], r["image_name"], r["class_label"]): r
        for r in csv.DictReader(open(path, encoding="utf-8"))
    }


def verdict(row: dict) -> str:
    return "CORRECT" if str(row.get("judged_correct", "")) == "1" else "INCORRECT"


def bq(label: str, text: str) -> list[str]:
    """Render `label: text` as a Markdown blockquote, prefixing every line
    (including ones from multi-paragraph template text) with '>' so the quote
    doesn't visually break partway through."""
    body_lines = text.split("\n")
    out = [f"> {label}: {body_lines[0]}"]
    for extra in body_lines[1:]:
        out.append(f"> {extra}" if extra.strip() else ">")
    return out


def render_cot_yaml(row: dict) -> str:
    lines = [f"**Method: cot_yaml**  (multi-turn, {len(split_step_qa(row['reasoning']))} CoT steps + classification)\n"]
    for i, (q, a) in enumerate(split_step_qa(row["reasoning"]), 1):
        lines += bq(f"Step {i} — Q", q)
        lines.append(">")
        lines += bq("A", a)
        lines.append("")
    lines += bq("Classification — Q", cls_question(row["task_type"], row.get("crop")))
    lines.append(">")
    lines += bq("A", row["prediction"])
    lines.append("")
    lines.append(f"**Verdict: {verdict(row)}**  (ground truth: {row['ground_truth']})")
    return "\n".join(lines)


def render_self_cot(row: dict) -> str:
    lines = [f"**Method: self_cot**  (2-turn: describe, then classify)\n"]
    lines += bq("Stage 1 — Q", desc_question(row["task_type"], row.get("crop")))
    lines.append(">")
    lines += bq("A", "*[not persisted — self_cot's stage-1 description was never saved to disk in "
                     "this pipeline run; only the final classification prediction below was recorded]*")
    lines.append("")
    lines += bq("Stage 2 — Q", cls_question(row["task_type"], row.get("crop")))
    lines.append(">")
    lines += bq("A", row["prediction"])
    lines.append("")
    lines.append(f"**Verdict: {verdict(row)}**  (ground truth: {row['ground_truth']})")
    return "\n".join(lines)


def render_desc_aug(row: dict) -> str:
    lines = [f"**Method: desc_aug**  (single-turn: pre-written class-level description used as context)\n"]
    lines += bq("Note", "*unlike self_cot/cot_yaml, this description is not generated from this "
                        f"specific image — it is a shared reference description for the "
                        f"'{row['class_label']}' class, reused as context for every image of that class.*")
    lines.append("")
    q = desc_aug_question(row["task_type"], row.get("crop"), row["lang_description"])
    lines += bq("Q", q)
    lines.append(">")
    lines += bq("A", row["prediction"])
    lines.append("")
    lines.append(f"**Verdict: {verdict(row)}**  (ground truth: {row['ground_truth']})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--cot_dir", default="outputs/cot_vision/")
    ap.add_argument("--step_dir", default="outputs/cot_step_analysis/")
    ap.add_argument("--desc_aug_dir", default="outputs/desc_augmented_vision/")
    ap.add_argument("--output_dir", default="outputs/verifier_analysis/")
    ap.add_argument("--rotation", default="0")
    ap.add_argument("--n_examples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    safe = args.model.replace("/", "_")
    swap_path = Path(args.step_dir) / "swap_order" / "ckpt_swap_judged.csv"
    self_cot_path = Path(args.cot_dir) / f"cot_cls_{safe}.csv"
    desc_aug_path = Path(args.desc_aug_dir) / f"aug_cls_all_{safe}.csv"
    out_path = Path(args.output_dir) / "sample_conversations.md"

    print(f"Loading cot_yaml (swap, rotation {args.rotation}): {swap_path}")
    cot_yaml = load_swap(swap_path, rotation=args.rotation)
    print(f"Loading self_cot: {self_cot_path}")
    self_cot = load_self_cot(self_cot_path)
    print(f"Loading desc_aug: {desc_aug_path}")
    desc_aug = load_desc_aug(desc_aug_path)

    common = set(cot_yaml) & set(self_cot) & set(desc_aug)
    print(f"3-way matched images: {len(common):,}")

    # Group by verdict pattern so we can hand-pick a diagnostically useful spread
    # rather than a random sample that might be all-agree / all-disagree.
    patterns: dict[tuple, list] = {}
    for k in common:
        pat = (verdict(cot_yaml[k]), verdict(self_cot[k]), verdict(desc_aug[k]))
        patterns.setdefault(pat, []).append(k)

    rng = random.Random(args.seed)
    wanted_patterns = [
        ("INCORRECT", "CORRECT", "CORRECT"),   # cot_yaml uniquely fails
        ("CORRECT", "INCORRECT", "INCORRECT"), # cot_yaml uniquely succeeds
        ("INCORRECT", "INCORRECT", "INCORRECT"),  # all three fail — shared blind spot
        ("CORRECT", "CORRECT", "CORRECT"),     # all three succeed — baseline case
    ]
    chosen: list[tuple] = []
    for pat in wanted_patterns:
        pool = patterns.get(pat, [])
        if pool:
            chosen.append(rng.choice(pool))
        if len(chosen) >= args.n_examples:
            break
    # top up with anything else if we didn't hit n_examples
    remaining = [k for k in common if k not in chosen]
    rng.shuffle(remaining)
    while len(chosen) < args.n_examples and remaining:
        chosen.append(remaining.pop())

    doc = ["# Sample Conversations — cot_yaml vs self_cot vs desc_aug\n",
           f"Model: {args.model}  |  {len(chosen)} examples, matched on the exact same image "
           f"across all three methods (3-way overlap: {len(common):,} images available).\n",
           "Each example shows the full, untruncated prompts and model responses copied "
           "directly from the underlying CSVs — nothing shortened or paraphrased.\n",
           "---\n"]

    for i, k in enumerate(chosen, 1):
        ds, basename, cls = k
        cy_row, sc_row, da_row = cot_yaml[k], self_cot[k], desc_aug[k]
        doc.append(f"## Example {i}: {ds} / {cls} / {basename}\n")
        doc.append(f"Verdicts — cot_yaml: **{verdict(cy_row)}**, "
                   f"self_cot: **{verdict(sc_row)}**, desc_aug: **{verdict(da_row)}**\n")
        doc.append(render_cot_yaml(cy_row) + "\n")
        doc.append(render_self_cot(sc_row) + "\n")
        doc.append(render_desc_aug(da_row) + "\n")
        doc.append("---\n")

    out_path.write_text("\n".join(doc))
    print(f"\nSaved: {out_path}  ({len(chosen)} examples)")


if __name__ == "__main__":
    main()
