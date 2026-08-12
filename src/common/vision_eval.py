"""Shared per-sample checkpoint + collation helpers for every classification
experiment in this repo — vision_baseline.py / desc_augmented_vision_v2.py
(lower/upper bound), self_cot.py / cot_yaml.py (chain-of-thought) — each
with a per-model output subfolder and cross-model collation, so a
limited-time remote job can resume exactly where it left off.
"""
from __future__ import annotations

import csv
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from src.common.metrics import apply_judge, aggregate_rouge, compute_classification_metrics, judged_summary
from src.common.progress import ProgressBar

try:
    from vllm import LLM
except ImportError:
    LLM = None  # type: ignore[assignment,misc]

CKPT_FIELDS = [
    "agml_dataset", "class_label", "task_type", "crop", "image_name",
    "prediction", "resolved_class", "ground_truth",
    "correct", "contains_match", "resolvable",
    "rouge1", "rouge2", "rougeL", "rouge_ss", "sem_sim",
    "reasoning",  # self_cot's own generated description / cot_yaml's step-by-step
                  # Q&A trail — empty string for vision_baseline.py and
                  # desc_augmented_vision_v2.py, which have no reasoning turn
    "judge_gt_hint",  # class_label + datasets.yaml class_aliases, see judge_gt_hint()
    "judged_correct", "judge_reasoning",  # filled in by the separate judge phase;
                                           # empty string until then
]

_SUMMARY_FIELDS = ["f1_macro", "accuracy", "contains_accuracy", "rouge_ss", "sem_sim", "judge_accuracy"]


def safe_model_name(model_id: str) -> str:
    return model_id.replace("/", "_").replace(" ", "_")


def ensure_judge_gt_hint(rows: list[dict]) -> int:
    """Backfill judge_gt_hint from class_label, in place, for any row where
    it's missing or blank — e.g. a checkpoint CSV written before
    judge_gt_hint existed as a column has no such key in its DictReader rows
    at all, and apply_judge()'s r.get(gt_key, "") silently resolves to ""
    for those, judging every prediction against a blank ground truth (which
    a judge will near-universally mark as no-match). Returns how many rows
    were backfilled, so callers can warn if it's a lot."""
    n = 0
    for r in rows:
        if not r.get("judge_gt_hint"):
            r["judge_gt_hint"] = r.get("class_label", "")
            n += 1
    return n


def judge_gt_hint(clean_label: str, aliases: list[str]) -> str:
    """Ground-truth string shown to the judge: the class label plus any
    datasets.yaml class_aliases (common/scientific/regional names for classes
    whose raw label is a cryptic code — species codes, cultivar IDs, nutrient
    element shorthand). Without this, a judge that only sees e.g. "SalKal" or
    "1121" as ground truth would have no way to recognize a correct
    prediction like "Russian thistle" or "Pusa Basmati 1121" as a match."""
    if not aliases:
        return clean_label
    return f"{clean_label} (also known as: {', '.join(aliases)})"


# Columns that have never moved/renamed across every CKPT_FIELDS revision
# this session — used by load_ckpt() to align rows from a stale-header file
# without needing to know exactly which past schema(s) it was written under.
_CKPT_FRONT_FIELDS = CKPT_FIELDS[: CKPT_FIELDS.index("sem_sim") + 1]
_CKPT_BACK_FIELDS = ["judged_correct", "judge_reasoning"]
_CKPT_MIDDLE_FIELDS = [f for f in CKPT_FIELDS if f not in _CKPT_FRONT_FIELDS and f not in _CKPT_BACK_FIELDS]


def load_ckpt(path: Path) -> tuple[list[dict], set[tuple[str, str, str]]]:
    """Load an existing checkpoint CSV. Returns (rows, seen-keys) so a resumed
    run can skip (dataset, class_label, image_name) triples already scored.

    Handles a stale on-disk header: append_ckpt_row() only writes a header
    for a brand-new file, so every time CKPT_FIELDS has gained a column
    (has happened repeatedly this session — judge_gt_hint, then reasoning)
    after a checkpoint file already existed, rows appended since are written
    in the CURRENT field order while line 1 still reflects an OLDER one. A
    plain csv.DictReader then zips the new, longer rows against the old,
    shorter header, silently shifting every value from the divergence point
    on — e.g. a class name landing in what reads as judged_correct,
    breaking int(r["judged_correct"]) deep in judged_summary(). A file can
    even mix rows from more than two schema generations if it predates
    multiple additions.

    Every past CKPT_FIELDS revision only ever inserted new columns into the
    same middle slot (between sem_sim and judged_correct) — the front
    columns (agml_dataset..sem_sim) and back columns (judged_correct,
    judge_reasoning) have stayed fixed all session — so each row can be
    aligned unambiguously from both ends using its own field COUNT, without
    needing to know exactly which historical schema(s) are mixed into the
    file. The file is then rewritten with a consistent, current header so
    future append_ckpt_row() calls stay aligned."""
    if not path.exists():
        return [], set()
    with open(path, newline="", encoding="utf-8") as f:
        raw_rows = list(csv.reader(f))
    if not raw_rows:
        return [], set()
    header, raw_rows = raw_rows[0], raw_rows[1:]
    n_front, n_back = len(_CKPT_FRONT_FIELDS), len(_CKPT_BACK_FIELDS)

    rows = []
    n_healed = 0
    for raw in raw_rows:
        n_middle = len(raw) - n_front - n_back
        if len(raw) == len(CKPT_FIELDS) and header == CKPT_FIELDS:
            row = dict(zip(CKPT_FIELDS, raw))
        elif 0 <= n_middle <= len(_CKPT_MIDDLE_FIELDS):
            # Middle fields were added in order (judge_gt_hint, then
            # reasoning inserted before it) — a row with fewer middle
            # values than today's schema is missing the most-recently-added
            # ones, so take the middle schema's own trailing slice, not its
            # front, to line values up with the fields that actually exist
            # in a row this short.
            middle = _CKPT_MIDDLE_FIELDS[-n_middle:] if n_middle else []
            fields = _CKPT_FRONT_FIELDS + middle + _CKPT_BACK_FIELDS
            row = dict(zip(fields, raw))
            for k in CKPT_FIELDS:
                row.setdefault(k, "")
            n_healed += 1
        else:
            raise ValueError(
                f"{path}: row has {len(raw)} fields — can't align to any known "
                f"CKPT_FIELDS schema (current has {len(CKPT_FIELDS)}): {raw[:5]}..."
            )
        rows.append(row)

    if n_healed:
        print(f"  [ckpt] {path.name}: healed {n_healed} row(s) written under a "
              f"stale header (schema drift) — rewriting with current schema.")
        rewrite_ckpt(path, rows)

    seen = {(r["agml_dataset"], r["class_label"], r["image_name"]) for r in rows}
    return rows, seen


def append_ckpt_row(path: Path, row: dict) -> None:
    """Flush one scored sample to the checkpoint CSV immediately — a kill
    mid-run loses at most the in-flight batch, not the whole model's progress."""
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CKPT_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CKPT_FIELDS})


# ── Judge phase ──────────────────────────────────────────────────────────────
#
# Judging is its own resumable phase, not part of the per-sample generation
# checkpoint loop: it needs a different model loaded (the judge, default
# google/gemma-4-26B-A4B-it, fixed across every baseline/model being
# evaluated), so it always runs after the classification model for this run
# has already been destroyed. rows_needing_judge() lets a resumed run only
# judge what a previous run didn't get to; rewrite_ckpt() persists the
# merged result (a full rewrite, not an append, since it's filling in
# columns on existing rows rather than adding new ones).
#
# run_judge_phase() is THE shared entry point every classification
# experiment (vision_baseline.py, desc_augmented_vision_v2.py, and any
# future one) calls — same judge model, same chunking/checkpoint/resume/
# rejudge/backfill behavior everywhere, so judge_accuracy is comparable
# across experiments and a fix here (like the blank-judge_gt_hint bug) only
# ever needs to happen in one place. Do not re-implement this per-script.

def _judge_row_key(row: dict) -> tuple[str, str, str]:
    return (row.get("agml_dataset", ""), row.get("class_label", ""), row.get("image_name", ""))


def rows_needing_judge(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not r.get("judged_correct")]


def merge_judged(rows: list[dict], judged_new: list[dict]) -> list[dict]:
    """Overlay judged_new's judged_correct/judge_reasoning onto the matching
    rows in `rows` (by dataset/class/image key), leaving every other row and
    every other column untouched."""
    new_map = {_judge_row_key(r): r for r in judged_new}
    merged = []
    for r in rows:
        j = new_map.get(_judge_row_key(r))
        if j is None:
            merged.append(r)
        else:
            merged.append({**r, "judged_correct": j.get("judged_correct", ""),
                            "judge_reasoning": j.get("judge_reasoning", "")})
    return merged


def rewrite_ckpt(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CKPT_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CKPT_FIELDS})


def _load_judge_llm(
    model_id: str,
    quantization: str | None = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    max_num_seqs: int = 64,
    enforce_eager: bool = False,
):
    print(f"\n[judge] Loading {model_id} via vLLM")
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        quantization=quantization,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=enforce_eager,
    )
    return llm, llm.get_tokenizer()


def _destroy_judge_llm(llm) -> None:
    import torch
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    del llm
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def run_judge_phase(rows: list[dict], ckpt_path: Path, args, prompts: dict) -> list[dict]:
    """Shared judge phase — see module-level note above.

    `args` (any argparse.Namespace-like object — vision_baseline.py,
    desc_augmented_vision_v2.py, self_cot.py, and cot_yaml.py all define
    these exact flags, and any new experiment reusing this function should
    too) must carry:
      judge_model, judge_quantize, judge_confidence_threshold,
      judge_chunk_size, rejudge, skip_judge, gpu_memory_utilization,
      max_model_len, max_num_seqs, enforce_eager

    `prompts` must carry judge_classification_template (and optionally
    judge_system_vision) from prompts.yaml.

    Every judge call's full input/output (system prompt, rendered prompt,
    untruncated raw response, gt/pred shown) is logged to
    <ckpt_path's directory>/judge_log.jsonl for QA — this is what caught the
    blank-judge_gt_hint bug; keep it on by default rather than making it
    opt-in.

    args.skip_judge (--skip_judge): skip this phase entirely — no judge
    model loaded, existing judged_correct/judge_reasoning left untouched.
    For preliminary/small-GPU runs where you want classification metrics
    (F1, ROUGE) without the extra judge model's memory/time cost. Resume
    later with a normal (non-skip) run to backfill judging.
    """
    if getattr(args, "skip_judge", False):
        print("  [judge] --skip_judge set — skipping judge phase entirely")
        return rows

    log_path = ckpt_path.parent / "judge_log.jsonl"

    if args.rejudge:
        for r in rows:
            r["judged_correct"] = ""
            r["judge_reasoning"] = ""

    n_backfilled = ensure_judge_gt_hint(rows)
    if n_backfilled:
        print(f"  [judge] backfilled judge_gt_hint (from class_label) on {n_backfilled} row(s) "
              f"missing it — likely a checkpoint written before that column existed")

    todo = rows_needing_judge(rows)
    if not todo:
        if rows:
            print(f"  [judge] all {len(rows)} row(s) already judged")
        return rows

    print(f"  [judge] {len(todo)} row(s) to judge ({len(rows) - len(todo)} already judged) "
          f"— loading {args.judge_model}")
    print(f"  [judge] logging full conversations to {log_path}")
    judge_llm, judge_tok = _load_judge_llm(
        args.judge_model, quantization=args.judge_quantize,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
    )

    chunk_size = max(args.judge_chunk_size, 1)
    pbar = ProgressBar(total=len(todo), desc="Judging")
    for i in range(0, len(todo), chunk_size):
        chunk = todo[i : i + chunk_size]
        judged_chunk = apply_judge(
            rows=chunk, llm=judge_llm, tokenizer=judge_tok,
            system_prompt=prompts.get("judge_system_vision", ""),
            judge_template=prompts["judge_classification_template"],
            gt_key="judge_gt_hint", pred_key="prediction",
            confidence_threshold=args.judge_confidence_threshold,
            log_path=log_path,
        )
        rows = merge_judged(rows, judged_chunk)
        rewrite_ckpt(ckpt_path, rows)
        pbar.update(len(chunk))
    pbar.finish()

    _destroy_judge_llm(judge_llm)
    del judge_tok
    print(f"  [judge] done — checkpoint rewritten")
    return rows


def compute_model_metrics(rows: list[dict]) -> dict:
    """Per-dataset, per-task-type, and overall F1 + ROUGE (+ judge accuracy, if
    the judge phase has run) for one model's rows."""
    by_ds: dict[str, list[dict]] = defaultdict(list)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ds[r["agml_dataset"]].append(r)
        by_task[r["task_type"]].append(r)

    def _summarize(group_rows: list[dict]) -> dict:
        refs = [r["ground_truth"] for r in group_rows]
        preds = [r.get("resolved_class") or "" for r in group_rows]
        m = compute_classification_metrics(refs, preds)
        m.update(aggregate_rouge(group_rows))
        m.update(judged_summary(group_rows))
        return m

    return {
        "overall":    _summarize(rows),
        "by_dataset": {ds: _summarize(rs) for ds, rs in by_ds.items()},
        "by_task":    {tt: _summarize(rs) for tt, rs in by_task.items()},
    }


def write_model_metrics(model_dir: Path, metrics: dict) -> None:
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def collate_models(output_dir: Path) -> dict[str, dict]:
    """Scan output_dir/<model>/ckpt_*.csv (one subfolder per model already
    run), recompute metrics per model, return {model_dir_name: metrics}."""
    if not output_dir.exists():
        return {}
    result = {}
    for model_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        ckpt_files = list(model_dir.glob("ckpt_*.csv"))
        if not ckpt_files:
            continue
        rows = list(csv.DictReader(open(ckpt_files[0], encoding="utf-8")))
        if not rows:
            continue
        result[model_dir.name] = compute_model_metrics(rows)
    return result


# ── Cross-script (baseline + desc-aug together) collation ──────────────────
#
# vision_baseline.py's own --collate and desc_augmented_vision_v2.py's own
# --collate (above) each summarize one script's output_dir in isolation —
# useful for a quick per-script progress check. Comparing the two conditions
# against each other (the actual lower-bound/upper-bound question) needs both
# directories at once; that combined view + the CI'd bar chart live in
# vision_baseline_collate.py, built on the two helpers below.

def load_all_rows(output_dir: Path) -> dict[str, list[dict]]:
    """{model_dir_name: [row, ...]} — every ckpt_*.csv under output_dir/<model>/."""
    if not output_dir.exists():
        return {}
    result = {}
    for model_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        ckpt_files = list(model_dir.glob("ckpt_*.csv"))
        if not ckpt_files:
            continue
        rows = list(csv.DictReader(open(ckpt_files[0], encoding="utf-8")))
        if rows:
            result[model_dir.name] = rows
    return result


def per_class_f1_values(rows: list[dict]) -> tuple[list[str], np.ndarray]:
    """(classes, per_class F1 array) — each class's own F1 from its own
    TP/FP/FN, computed via resolved_class vs ground_truth. Shared by
    bootstrap_f1_ci (resamples this array) and the per-class percentile/
    consistency plot (shows this array's raw spread directly, no resampling)."""
    refs = [r["ground_truth"] for r in rows]
    preds = [r.get("resolved_class") or "" for r in rows]
    classes = sorted(set(refs))
    if not classes:
        return [], np.array([])
    per_class = f1_score(refs, preds, average=None, labels=classes, zero_division=0)
    return classes, per_class


def bootstrap_f1_ci(rows: list[dict], n_boot: int = 2000, seed: int = 42) -> dict:
    """Macro-F1 point estimate + 95% CI, bootstrap-resampled over CLASSES (not
    samples) — each class already has its own well-defined F1 (from its own
    TP/FP/FN), and macro-F1 is just their mean, so resampling which classes
    are in the average is the natural unit of uncertainty here, matching "CI
    based on the classes" rather than per-image noise."""
    classes, per_class = per_class_f1_values(rows)
    if not classes:
        return {"f1_macro": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_classes": 0, "n": len(rows)}

    point = float(np.mean(per_class))
    if len(classes) < 2:
        return {"f1_macro": round(point, 4), "ci_low": round(point, 4),
                "ci_high": round(point, 4), "n_classes": len(classes), "n": len(rows)}

    rng = np.random.default_rng(seed)
    idx = np.arange(len(classes))
    boot_means = np.array([
        per_class[rng.choice(idx, size=len(classes), replace=True)].mean()
        for _ in range(n_boot)
    ])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return {
        "f1_macro": round(point, 4),
        "ci_low": round(float(ci_low), 4),
        "ci_high": round(float(ci_high), 4),
        "n_classes": len(classes),
        "n": len(rows),
    }


def compute_condition_stats(rows: list[dict], n_boot: int = 2000, seed: int = 42) -> dict[str, dict]:
    """{"overall": {...}, "<task_type>": {...}, ...} — bootstrap_f1_ci scoped
    to all rows and to each task_type subset."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task_type"]].append(r)
    stats = {"overall": bootstrap_f1_ci(rows, n_boot, seed)}
    for tt, trows in by_task.items():
        stats[tt] = bootstrap_f1_ci(trows, n_boot, seed)
    return stats


def write_collated(output_dir: Path, collated: dict[str, dict]) -> None:
    (output_dir / "collated_metrics.json").write_text(json.dumps(collated, indent=2))

    fieldnames = ["model", "scope", "key", "n_total", *_SUMMARY_FIELDS]
    with open(output_dir / "collated_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model, metrics in collated.items():
            def _row(scope: str, key: str, m: dict) -> dict:
                row = {"model": model, "scope": scope, "key": key, "n_total": m.get("n_total")}
                row.update({k: m.get(k) for k in _SUMMARY_FIELDS})
                return row
            writer.writerow(_row("overall", "", metrics["overall"]))
            for tt, tm in metrics["by_task"].items():
                writer.writerow(_row("task", tt, tm))
            for ds, dm in metrics["by_dataset"].items():
                writer.writerow(_row("dataset", ds, dm))
    print(f"  Saved: {output_dir / 'collated_summary.csv'}")
    print(f"  Saved: {output_dir / 'collated_metrics.json'}")
