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
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from src.common.dataset_config import excluded_class_pairs
from src.common.metrics import apply_judge, aggregate_rouge, compute_classification_metrics, judged_summary
from src.common.progress import ProgressBar

try:
    from vllm import LLM
except ImportError:
    LLM = None  # type: ignore[assignment,misc]

_DATASETS_CONFIG_PATH = Path(__file__).parent.parent / "config" / "datasets.yaml"
_EXCLUDED_CLASS_PAIRS_CACHE: set[tuple[str, str]] | None = None


def _cached_excluded_class_pairs() -> set[tuple[str, str]]:
    """Memoized excluded_class_pairs() over the default datasets.yaml —
    load_ckpt() is called once per checkpoint file across a run/collation
    sweep (dozens of times for a full outputs_final regeneration), and the
    exclusion set never changes mid-process, so re-parsing the whole YAML
    every call would be pure waste."""
    global _EXCLUDED_CLASS_PAIRS_CACHE
    if _EXCLUDED_CLASS_PAIRS_CACHE is None:
        _EXCLUDED_CLASS_PAIRS_CACHE = excluded_class_pairs(_DATASETS_CONFIG_PATH)
    return _EXCLUDED_CLASS_PAIRS_CACHE

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


def _split_front_middle_back(
    fields: list[str], header: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """Longest common prefix/suffix between the CURRENT schema (`fields`) and
    an on-disk `header` that may reflect an older revision of the same
    checkpoint schema. Whatever's left of `fields` in between is the
    "middle" — the span where past schema changes have actually inserted
    columns (e.g. CKPT_FIELDS gained judge_gt_hint, then reasoning, both
    slotted in between the front columns and the trailing judge_* columns,
    which is why old and new rows still share this same stable prefix/
    suffix). Computed fresh from whatever header is actually in the file
    (not a hardcoded list of past field names), so this generalizes to any
    fields/CKPT_FIELDS-shaped schema, not just the specific history of
    CKPT_FIELDS itself."""
    n = min(len(fields), len(header))
    front = 0
    while front < n and fields[front] == header[front]:
        front += 1
    back = 0
    while back < n - front and fields[len(fields) - 1 - back] == header[len(header) - 1 - back]:
        back += 1
    front_fields = fields[:front]
    back_fields = fields[len(fields) - back:] if back else []
    middle_fields = fields[front: len(fields) - back] if back else fields[front:]
    return front_fields, middle_fields, back_fields


def load_ckpt(
    path: Path, fields: list[str] = CKPT_FIELDS
) -> tuple[list[dict], set[tuple[str, str, str]]]:
    """Load an existing checkpoint CSV. Returns (rows, seen-keys) so a resumed
    run can skip (dataset, class_label, image_name) triples already scored.

    `fields` defaults to CKPT_FIELDS (the single-final-prediction row shape
    used by vision_baseline.py / desc_augmented_vision_v2.py / self_cot.py /
    cot_yaml.py) but any checkpoint using this same append/rewrite machinery
    with a different row shape (e.g. cot_yaml.py's per-step STEP_CKPT_FIELDS)
    can pass its own field list here.

    Handles a stale on-disk header: append_ckpt_row() only writes a header
    for a brand-new file, so every time a schema gains a column (has
    happened repeatedly this session for CKPT_FIELDS — judge_gt_hint, then
    reasoning) after a checkpoint file already existed, rows appended since
    are written in the CURRENT field order while line 1 still reflects an
    OLDER one. A plain csv.DictReader then zips the new, longer rows against
    the old, shorter header, silently shifting every value from the
    divergence point on — e.g. a class name landing in what reads as
    judged_correct, breaking int(r["judged_correct"]) deep in
    judged_summary(). A file can even mix rows from more than two schema
    generations if it predates multiple additions.

    Each row is aligned by its own field COUNT against the stable front/back
    columns shared between `fields` and the file's actual header (see
    _split_front_middle_back) — no need to know exactly which historical
    schema(s) are mixed into the file. The file is then rewritten with a
    consistent, current header so future append_ckpt_row() calls stay
    aligned."""
    if not path.exists():
        return [], set()
    with open(path, newline="", encoding="utf-8") as f:
        raw_rows = list(csv.reader(f))
    if not raw_rows:
        return [], set()
    header, raw_rows = raw_rows[0], raw_rows[1:]
    front, middle, back = _split_front_middle_back(fields, header)
    n_front, n_back = len(front), len(back)

    rows = []
    n_healed = 0
    for raw in raw_rows:
        n_middle = len(raw) - n_front - n_back
        if len(raw) == len(fields) and header == fields:
            row = dict(zip(fields, raw))
        elif 0 <= n_middle <= len(middle):
            # A row with fewer middle values than today's schema is missing
            # the most-recently-added ones — take the middle schema's own
            # TRAILING slice (not its front), since every past insertion has
            # landed new columns before the existing middle ones, not after.
            row_fields = front + (middle[-n_middle:] if n_middle else []) + back
            row = dict(zip(row_fields, raw))
            for k in fields:
                row.setdefault(k, "")
            n_healed += 1
        else:
            raise ValueError(
                f"{path}: row has {len(raw)} fields — can't align to any known "
                f"schema (current has {len(fields)}): {raw[:5]}..."
            )
        rows.append(row)

    if n_healed:
        print(f"  [ckpt] {path.name}: healed {n_healed} row(s) written under a "
              f"stale header (schema drift) — rewriting with current schema.")
        rewrite_ckpt(path, rows, fields=fields)

    excluded = _cached_excluded_class_pairs()
    if excluded:
        n_before = len(rows)
        rows = [r for r in rows if (r.get("agml_dataset"), r.get("class_label")) not in excluded]
        if len(rows) != n_before:
            print(f"  [ckpt] {path.name}: dropped {n_before - len(rows)} row(s) for classes now "
                  f"marked excluded_classes in datasets.yaml.")

    seen = {(r["agml_dataset"], r["class_label"], r["image_name"]) for r in rows}
    return rows, seen


def append_ckpt_row(path: Path, row: dict, fields: list[str] = CKPT_FIELDS) -> None:
    """Flush one scored sample to the checkpoint CSV immediately — a kill
    mid-run loses at most the in-flight batch, not the whole model's progress."""
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def archive_ckpt_if_rerun(
    ckpt_path: Path, rerun: bool, extra_logs: tuple[str, ...] = ("judge_log.jsonl",),
) -> None:
    """--rerun support: load_ckpt()'s resume-by-key behavior is normally a
    feature (survives a kill mid-run) but becomes a liability the one time
    you actually WANT every row regenerated — e.g. after fixing a bug in how
    prompts were built (silently-enabled Qwen "thinking" mode truncating
    classification answers) that corrupted predictions already checkpointed
    under the old code. Just rerunning the same command wouldn't help: every
    row already present in the checkpoint would be silently kept, bug and
    all, since load_ckpt has no way to tell a corrupted row from a finished
    one.

    Moves ckpt_path (and each of `extra_logs` present in its directory —
    default just judge_log.jsonl, so a fresh run's judge conversations
    don't mix into the old log) aside with a timestamp — never overwrites a
    previous archive — rather than deleting, so old data is always
    recoverable. No-op if `rerun` is false or there's nothing at ckpt_path
    yet. Pass extra_logs=("judge_log.jsonl", "verifier_log_belief.jsonl")
    (etc.) for a script with its OWN raw per-call log beyond the checkpoint
    + judge log — e.g. verifier_cot.py's verifier_log_<baseline>.jsonl,
    which used to just keep appending across every --rerun forever (old and
    new run entries silently concatenated in one file, with no way to tell
    them apart short of diffing prompt wording) since nothing ever archived
    it."""
    if not rerun or not ckpt_path.exists():
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = ckpt_path.parent / f"{ckpt_path.stem}.{ts}.bak.csv"
    ckpt_path.rename(backup)
    print(f"  [rerun] moved existing checkpoint aside: {backup.name}")

    for name in extra_logs:
        log_path = ckpt_path.parent / name
        if log_path.exists():
            stem, _, ext = name.rpartition(".")
            log_backup = log_path.parent / f"{stem}.{ts}.bak.{ext}"
            log_path.rename(log_backup)
            print(f"  [rerun] moved existing log aside: {log_backup.name}")


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

def _judge_row_key(row: dict) -> tuple[str, str, str, str, str, str]:
    """(dataset, class, image[, step_idx, rotation, round]) — the last three
    are "" for every CKPT_FIELDS row (one prediction per image). step_idx/
    rotation are only meaningful for cot_yaml.py's STEP_CKPT_FIELDS rows
    (multiple predictions per image, one per chain position/rotation);
    round is only meaningful for verifier_cot.py's VERIFIER_STEP_FIELDS rows
    (multiple predictions per image, one per round). Without `round` here,
    every round of the same image shared one key, so merge_judged's
    last-write-wins dict silently overwrote every earlier round's real
    judged_correct with whichever round happened to be judged last for that
    image — a real bug (each round WAS independently judged, correctly;
    only the write-back collapsed them). Because verifier_cot.py writes rows
    image-major/round-ascending, the round that "won" was always the
    image's own final round, so any chart reading only is_final_round=1
    rows was accidentally unaffected — but anything reading per-round
    judged_correct (e.g. a belief-vs-round judged-accuracy plot) was not."""
    return (row.get("agml_dataset", ""), row.get("class_label", ""), row.get("image_name", ""),
            row.get("round", ""),
            str(row.get("step_idx", "")), str(row.get("rotation", "")))


def rows_needing_judge(rows: list[dict], judge_filter=None) -> list[dict]:
    """`judge_filter`, if given, is an additional predicate a row must pass
    to be judged — e.g. verifier_cot.py's --judge_scope first_and_final
    restricts judging to round==1 / is_final_round==1 rows only, leaving
    every intermediate round's judged_correct blank (never judged) to
    save judge compute when only a round-1-vs-final ablation is needed,
    not a full round-by-round trend. Rows a filter excludes are simply
    left unjudged (blank judged_correct), same as any other never-judged
    row — no other caller passes this, so every existing call site's
    behavior (judge everything not yet judged) is unchanged."""
    todo = [r for r in rows if not r.get("judged_correct")]
    if judge_filter is not None:
        todo = [r for r in todo if judge_filter(r)]
    return todo


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


def rewrite_ckpt(path: Path, rows: list[dict], fields: list[str] = CKPT_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def _load_judge_llm(
    model_id: str,
    quantization: str | None = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    max_num_seqs: int = 64,
    enforce_eager: bool = False,
):
    from src.common.vlm_load import _best_supported_dtype
    dtype = _best_supported_dtype()
    print(f"\n[judge] Loading {model_id} via vLLM (dtype={dtype})")
    llm = LLM(
        model=model_id,
        dtype=dtype,
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


def run_judge_phase(
    rows: list[dict], ckpt_path: Path, args, prompts: dict, fields: list[str] = CKPT_FIELDS,
    judge_filter=None,
) -> list[dict]:
    """Shared judge phase — see module-level note above.

    judge_filter: passed straight through to rows_needing_judge — an
    additional predicate restricting which rows get judged this call
    (everything else is left with a blank judged_correct, same as any
    other not-yet-judged row). None (default) judges every row missing a
    verdict, unchanged from every existing caller's behavior.

    `args` (any argparse.Namespace-like object — vision_baseline.py,
    desc_augmented_vision_v2.py, self_cot.py, and cot_yaml.py all define
    these exact flags, and any new experiment reusing this function should
    too) must carry:
      judge_model, judge_quantize, judge_confidence_threshold,
      judge_chunk_size, rejudge, skip_judge, gpu_memory_utilization,
      max_model_len, max_num_seqs, enforce_eager

    `prompts` must carry judge_classification_template (and optionally
    judge_system_vision) from prompts.yaml.

    `fields` is the row schema to checkpoint with — defaults to CKPT_FIELDS,
    override for a differently-shaped checkpoint (e.g. cot_yaml.py's
    per-step STEP_CKPT_FIELDS) so the SAME judge implementation (same judge
    model, same prompts, same confidence threshold, same logging) scores
    every classification experiment in this repo, not a reimplementation
    per row shape.

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

    todo = rows_needing_judge(rows, judge_filter=judge_filter)
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
        rewrite_ckpt(ckpt_path, rows, fields=fields)
        pbar.update(len(chunk))
    pbar.finish()

    _destroy_judge_llm(judge_llm)
    del judge_tok
    print(f"  [judge] done — checkpoint rewritten")
    return rows


def compute_step_metrics(rows: list[dict], group_keys: tuple[str, ...]) -> dict[tuple, dict]:
    """Group step-level rows (cot_yaml.py's STEP_CKPT_FIELDS shape — multiple
    rows per image, one per chain position) by `group_keys` — e.g.
    ("agml_dataset", "step_idx") for a per-step F1 curve, or
    ("agml_dataset", "rotation", "step_idx") for the --swap_datasets sweep —
    and compute the SAME F1/ROUGE/judge metrics compute_model_metrics uses
    for single-prediction rows. Reuses those exact scoring functions rather
    than a step-analysis-specific reimplementation, so results stay
    comparable with vision_baseline.py / desc_augmented_vision_v2.py /
    self_cot.py / cot_yaml.py's own final-answer metrics."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k, "") for k in group_keys)].append(r)

    def _summarize(group_rows: list[dict]) -> dict:
        refs = [r["ground_truth"] for r in group_rows]
        preds = [r.get("resolved_class") or "" for r in group_rows]
        m = compute_classification_metrics(refs, preds)
        m.update(aggregate_rouge(group_rows))
        m.update(judged_summary(group_rows))
        return m

    return {key: _summarize(rs) for key, rs in groups.items()}


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
    """Scan output_dir/<model>/ckpt_<model>.csv (one subfolder per model
    already run), recompute metrics per model, return {model_dir_name: metrics}."""
    if not output_dir.exists():
        return {}
    result = {}
    for model_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        ckpt_path = model_dir / f"ckpt_{model_dir.name}.csv"
        rows, _ = load_ckpt(ckpt_path)
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
    """{model_dir_name: [row, ...]} — ckpt_<model>.csv under output_dir/<model>/,
    loaded through load_ckpt() (not a raw csv.DictReader) so a stale-header
    checkpoint (schema drift — see load_ckpt's docstring) gets healed here
    too rather than silently misreading judged_correct as a class-name
    string. Previously used model_dir.glob("ckpt_*.csv")[0], which — now
    that --rerun archives old checkpoints as ckpt_<model>.<timestamp>.bak.csv
    right alongside the live one — could match and load a STALE BACKUP
    instead of the current file; the exact expected filename sidesteps that
    too."""
    if not output_dir.exists():
        return {}
    result = {}
    for model_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        ckpt_path = model_dir / f"ckpt_{model_dir.name}.csv"
        rows, _ = load_ckpt(ckpt_path)
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
    based on the classes" rather than per-image noise.

    This is the STRICT metric — resolved_class vs ground_truth exact match,
    same as compute_classification_metrics's f1_macro. It can't credit a
    correct-but-differently-phrased prediction (e.g. a full descriptive
    sentence that never resolved to a bare class name) at all, which is
    exactly the gap the judge exists to close — see bootstrap_judge_f1_ci
    for the judge-scored counterpart, generally the more meaningful number
    for a headline comparison."""
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


def per_class_judge_f1_values(rows: list[dict]) -> tuple[list[str], np.ndarray]:
    """(classes, per_class F1 array), judge-scored — same TP/FP/FN-per-class
    construction as per_class_f1_values, but a row counts as a match
    whenever the judge said so (judged_correct), falling back to
    resolved_class only when the judge disagreed — identical prediction
    logic to judged_summary()'s judge_f1_macro (metrics.py), just exposed
    per-class instead of pre-averaged so bootstrap_judge_f1_ci can resample
    it exactly like the strict version does. Rows without a judged_correct
    value (judge phase skipped/not yet run) are excluded, same as
    judged_summary()."""
    judged = [r for r in rows if r.get("judged_correct", "") != ""]
    if not judged:
        return [], np.array([])
    jc = [int(r["judged_correct"]) for r in judged]
    refs = [r["class_label"] for r in judged]
    preds = [r["class_label"] if jc[i] else (r.get("resolved_class") or "") for i, r in enumerate(judged)]
    classes = sorted(set(refs))
    if not classes:
        return [], np.array([])
    per_class = f1_score(refs, preds, average=None, labels=classes, zero_division=0)
    return classes, per_class


def bootstrap_judge_f1_ci(rows: list[dict], n_boot: int = 2000, seed: int = 42) -> dict:
    """Judge-scored counterpart to bootstrap_f1_ci — same class-resampling
    CI construction, built on per_class_judge_f1_values instead of the
    strict per_class_f1_values. Falls back to all-zero (n_classes=0) if the
    judge phase hasn't run on these rows, same shape as the strict version's
    empty-case so callers can treat them uniformly."""
    classes, per_class = per_class_judge_f1_values(rows)
    if not classes:
        return {"judge_f1_macro": 0.0, "judge_ci_low": 0.0, "judge_ci_high": 0.0,
                "judge_n_classes": 0, "n": len(rows)}

    point = float(np.mean(per_class))
    if len(classes) < 2:
        return {"judge_f1_macro": round(point, 4), "judge_ci_low": round(point, 4),
                "judge_ci_high": round(point, 4), "judge_n_classes": len(classes), "n": len(rows)}

    rng = np.random.default_rng(seed)
    idx = np.arange(len(classes))
    boot_means = np.array([
        per_class[rng.choice(idx, size=len(classes), replace=True)].mean()
        for _ in range(n_boot)
    ])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return {
        "judge_f1_macro": round(point, 4),
        "judge_ci_low": round(float(ci_low), 4),
        "judge_ci_high": round(float(ci_high), 4),
        "judge_n_classes": len(classes),
        "n": len(rows),
    }


def per_class_judge_f1_delta(baseline_rows: list[dict], aug_rows: list[dict]) -> tuple[list[str], np.ndarray]:
    """Per-class (aug - baseline) judge-F1 delta, PAIRED on classes judged in
    both conditions. This is what bootstrap_judge_f1_delta_ci resamples —
    treating "does description-augmentation help this class" as the
    resampling unit, rather than the naive (and statistically invalid)
    shortcut of just subtracting two independently-bootstrapped CIs from
    each other."""
    base_classes, base_f1 = per_class_judge_f1_values(baseline_rows)
    aug_classes, aug_f1 = per_class_judge_f1_values(aug_rows)
    base_map = dict(zip(base_classes, base_f1))
    aug_map = dict(zip(aug_classes, aug_f1))
    common = sorted(set(base_map) & set(aug_map))
    if not common:
        return [], np.array([])
    deltas = np.array([aug_map[c] - base_map[c] for c in common])
    return common, deltas


def bootstrap_judge_f1_delta_ci(
    baseline_rows: list[dict], aug_rows: list[dict], n_boot: int = 2000, seed: int = 42
) -> dict:
    """Point estimate + 95% CI for the (aug - baseline) judge-F1 delta,
    bootstrap-resampled over the paired per-class deltas (see
    per_class_judge_f1_delta) — same class-resampling logic as
    bootstrap_judge_f1_ci, just resampling deltas instead of raw scores."""
    classes, deltas = per_class_judge_f1_delta(baseline_rows, aug_rows)
    if not classes:
        return {"delta_f1_macro": 0.0, "delta_ci_low": 0.0, "delta_ci_high": 0.0, "delta_n_classes": 0}

    point = float(np.mean(deltas))
    if len(classes) < 2:
        return {"delta_f1_macro": round(point, 4), "delta_ci_low": round(point, 4),
                "delta_ci_high": round(point, 4), "delta_n_classes": len(classes)}

    rng = np.random.default_rng(seed)
    idx = np.arange(len(classes))
    boot_means = np.array([
        deltas[rng.choice(idx, size=len(classes), replace=True)].mean()
        for _ in range(n_boot)
    ])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return {
        "delta_f1_macro": round(point, 4),
        "delta_ci_low": round(float(ci_low), 4),
        "delta_ci_high": round(float(ci_high), 4),
        "delta_n_classes": len(classes),
    }


def compute_delta_stats(
    baseline_rows: list[dict], aug_rows: list[dict], n_boot: int = 2000, seed: int = 42
) -> dict[str, dict]:
    """{"overall": {...}, "<task_type>": {...}, ...} — bootstrap_judge_f1_delta_ci
    scoped to all rows and to each task_type subset present in EITHER
    condition (a scope missing from one side just yields no common classes,
    handled gracefully by bootstrap_judge_f1_delta_ci's empty-case)."""
    base_by_task: dict[str, list[dict]] = defaultdict(list)
    for r in baseline_rows:
        base_by_task[r["task_type"]].append(r)
    aug_by_task: dict[str, list[dict]] = defaultdict(list)
    for r in aug_rows:
        aug_by_task[r["task_type"]].append(r)

    task_types = set(base_by_task) | set(aug_by_task)
    stats = {"overall": bootstrap_judge_f1_delta_ci(baseline_rows, aug_rows, n_boot, seed)}
    for tt in task_types:
        stats[tt] = bootstrap_judge_f1_delta_ci(base_by_task.get(tt, []), aug_by_task.get(tt, []), n_boot, seed)
    return stats


def compute_condition_stats(rows: list[dict], n_boot: int = 2000, seed: int = 42) -> dict[str, dict]:
    """{"overall": {...}, "<task_type>": {...}, ...} — bootstrap_f1_ci AND
    bootstrap_judge_f1_ci scoped to all rows and to each task_type subset,
    merged into one dict per scope (strict f1_macro/ci_low/ci_high alongside
    judge_f1_macro/judge_ci_low/judge_ci_high) so callers can plot either
    without recomputing."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task_type"]].append(r)

    def _both(group_rows: list[dict]) -> dict:
        merged = bootstrap_f1_ci(group_rows, n_boot, seed)
        merged.update(bootstrap_judge_f1_ci(group_rows, n_boot, seed))
        return merged

    stats = {"overall": _both(rows)}
    for tt, trows in by_task.items():
        stats[tt] = _both(trows)
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
