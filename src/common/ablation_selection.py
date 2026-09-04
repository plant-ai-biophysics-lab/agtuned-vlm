"""Ablation stage, "selection" step (Q2/Q3) — reconstructs per-candidate-pool
selection arms (first candidate, random, majority vote, oracle-in-pool) from
an existing verifier_cot.py / detection_verifier.py run's raw
verifier_log.jsonl.

The per-row checkpoint CSV (VERIFIER_CKPT_FIELDS, src/verifier_cot.py) only
keeps the tournament's WINNING candidate + aggregate top1_win_rate/
n_comparisons — it never stores the individual K candidate texts. Every arm
below needs the full pool, so it's rebuilt from the verifier_log.jsonl
sidecar (call_type == "candidate" rows), joined back to the checkpoint CSV
by (agml_dataset, class_label, image_name).

The verifier-selected arm itself is read straight off the checkpoint (no
reconstruction needed — it's exactly what the paper already reports).

Arms are scored by direct resolved-class match against ground truth, not the
LLM judge — re-judging every reconstructed arm would need a fresh GPU judge
pass. This mirrors the same direct-match metric vision_baseline.py /
desc_augmented_vision_v2.py already report alongside judged F1 (their
`correct` column), not the judge's own score.
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from src.common.dataset_config import datasets_by_name
from src.common.metrics import resolve_class
from src.majority_vote import plurality_vote

RowKey = tuple[str, str, str]  # (agml_dataset, class_label, image_name)

ARMS_NEEDING_POOL = ("first", "majority", "random", "oracle_in_pool")


def load_candidate_pools(verifier_log_path: Path) -> dict[RowKey, list[str]]:
    """{(agml_dataset, class_label, image_name): [raw_text, ...]} ordered by
    candidate_index. Only call_type == "candidate" rows are used —
    tournament_pair rows are the pairwise comparisons, not needed here.

    Deliberately keeps the FULL raw_text, not extracted_answer: the paper
    pipeline itself resolves classes against the whole winning trajectory
    (verifier_cot.py::run_group sets `prediction = pools[i][tr.top1]`, the
    raw candidate text, and _result_row calls
    `resolve_class(result["prediction"], ds.clean_classes)` on that full
    text — not on the <answer> tag alone). Resolving against the shorter
    extracted_answer instead measurably disagrees with the pipeline's own
    resolution (verified: it broke the oracle_in_pool >= verifier accuracy
    invariant), so every arm here must resolve against raw_text to stay
    comparable to the paper's own reported numbers.
    """
    pools: dict[RowKey, dict[int, str]] = defaultdict(dict)
    with open(verifier_log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("call_type") != "candidate":
                continue
            key = (rec["agml_dataset"], rec["class_label"], rec["image_name"])
            pools[key][rec["candidate_index"]] = rec.get("raw_text", "")
    return {key: [texts[i] for i in sorted(texts)] for key, texts in pools.items()}


def resolve_pool(raw_answers: list[str], classes: list[str]) -> list[str | None]:
    return [resolve_class(a, classes) for a in raw_answers]


def first_candidate(resolved: list[str | None]) -> str | None:
    return resolved[0] if resolved else None


def random_candidate(resolved: list[str | None], rng: random.Random) -> str | None:
    return rng.choice(resolved) if resolved else None


def majority_vote_candidate(resolved: list[str | None]) -> str | None:
    pick, _frac = plurality_vote(resolved)
    return pick


def oracle_in_pool(resolved: list[str | None], ground_truth: str) -> bool:
    gt = ground_truth.strip().lower()
    return any(r is not None and r.strip().lower() == gt for r in resolved)


def _is_correct(pick: str | None, ground_truth: str) -> int:
    return int(pick is not None and pick.strip().lower() == ground_truth.strip().lower())


def score_run(
    ckpt_csv: Path,
    verifier_log_jsonl: Path,
    datasets_cfg_path: Path,
    model_label: str,
    condition_label: str,
    n_random_seeds: int = 5,
) -> list[dict]:
    """One row per (agml_dataset, class_label, image_name) x arm.

    Every row carries: model, condition (e.g. the source run's directory
    name, used as the K/config label), agml_dataset, class_label, image_name,
    task_type, pool_size, arm, correct (0/1, or a [0,1] average for "random",
    which is averaged over n_random_seeds fixed seeds for stability), and
    n_calls (a cost proxy: generation calls + comparison calls actually
    spent producing this pool).

    Rows whose pool is missing or incomplete (candidate log truncated by an
    interrupted run) are skipped for the pool-dependent arms but the
    verifier arm — read straight from the checkpoint, no pool needed — is
    still emitted, so a partial log never silently drops the one arm that
    doesn't need it.
    """
    ds_cfg = datasets_by_name(datasets_cfg_path)
    pools = load_candidate_pools(verifier_log_jsonl)
    out: list[dict] = []

    with open(ckpt_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["agml_dataset"], row["class_label"], row["image_name"])
            gt = row["ground_truth"] or row["class_label"]
            pool_size = int(row["pool_size"])
            n_comparisons = int(row["n_comparisons"])
            base = {
                "model": model_label, "condition": condition_label,
                "agml_dataset": key[0], "class_label": key[1], "image_name": key[2],
                "task_type": row["task_type"], "pool_size": pool_size,
            }

            out.append({**base, "arm": "verifier",
                        "correct": int(row["correct"]),
                        "n_calls": pool_size + n_comparisons})

            pool = pools.get(key)
            if pool is None or len(pool) != pool_size:
                continue

            ds = ds_cfg.get(key[0])
            if ds is None:
                continue
            classes = ds.clean_classes
            resolved = resolve_pool(pool, classes)

            out.append({**base, "arm": "first",
                        "correct": _is_correct(first_candidate(resolved), gt),
                        "n_calls": pool_size})
            out.append({**base, "arm": "majority",
                        "correct": _is_correct(majority_vote_candidate(resolved), gt),
                        "n_calls": pool_size})

            rand_vals = [
                _is_correct(random_candidate(resolved, random.Random(seed)), gt)
                for seed in range(n_random_seeds)
            ]
            out.append({**base, "arm": "random",
                        "correct": sum(rand_vals) / len(rand_vals),
                        "n_calls": pool_size})

            out.append({**base, "arm": "oracle_in_pool",
                        "correct": int(oracle_in_pool(resolved, gt)),
                        "n_calls": pool_size})

    return out


def aggregate_arms(rows: list[dict]) -> list[dict]:
    """[{model, condition, arm, pool_size, task_type, n, accuracy, mean_n_calls}, ...]
    grouped by (model, condition, arm, task_type) and again with task_type="ALL"."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["condition"], r["arm"], r["task_type"])].append(r)
        groups[(r["model"], r["condition"], r["arm"], "ALL")].append(r)

    out = []
    for (model, condition, arm, task_type), grp in groups.items():
        pool_sizes = {r["pool_size"] for r in grp}
        out.append({
            "model": model, "condition": condition, "arm": arm, "task_type": task_type,
            "pool_size": pool_sizes.pop() if len(pool_sizes) == 1 else max(pool_sizes),
            "n": len(grp),
            "accuracy": sum(r["correct"] for r in grp) / len(grp),
            "mean_n_calls": sum(r["n_calls"] for r in grp) / len(grp),
        })
    return out
