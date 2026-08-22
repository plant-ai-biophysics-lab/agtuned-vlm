"""
Analyse self_cot and cot_yaml reasoning to find linguistic / trajectory signals
that predict correctness. Output feeds verifier design (what to flag, when).

Inputs:
  outputs/cot_vision/cot_cls_<model>.csv        final answers, self_cot + cot_yaml
  outputs/cot_step_analysis/ckpt_judged.csv     cot_yaml intermediate steps

Outputs:
  outputs/verifier_analysis/verifier_analysis.json
  outputs/verifier_analysis/*.png

Usage:
  uv run python src/verifier_analysis.py --model google/gemma-4-26B-A4B-it
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# ── Marker families ───────────────────────────────────────────────────────────
# Lowercase substrings. A row "has" the marker if any phrase appears.

MARKERS: dict[str, list[str]] = {
    "abstention": [
        "impossible to", "not possible to", "cannot provide", "cannot be",
        "i cannot", "unable to", "cannot definitively", "cannot make",
        "no definitive", "cannot determine", "impossible to determine",
    ],
    "hedge_soft": [
        "most likely", "appears to be", "suggests", "may be", "might be",
        "could be", "likely a", "probably", "possibly", "consistent with",
    ],
    "hedge_strong": [
        "speculative", "highly speculative", "cannot be certain",
        "without further", "would require", "requires laboratory",
        "microscopic analysis", "laboratory analysis", "insufficient",
    ],
    "disclaimer": [
        "i must preface", "professional disclaimer", "i must emphasize",
        "must lead with", "should not be used", "professionally honest",
        "as an agricultural scientist, i must", "important caveat",
    ],
    "image_quality_complaint": [
        "low resolution", "low in resolution", "out-of-focus", "out of focus",
        "blurry", "image quality", "extremely limited visual",
        "poor quality", "lacks depth of field", "highly magnified",
    ],
    "differential": [
        "however", "alternatively", "could also", "differential",
        "ranked", "on the other hand", "in contrast", "rather than",
    ],
    "commitment": [
        "characteristic of", "diagnostic of", "clearly", "definitively",
        "pathognomonic", "classic symptom", "hallmark", "unmistakab",
    ],
    "negative_finding": [
        "no visible", "no clear", "there are no", "absent", "no evidence of",
        "rule out", "no signs of",
    ],
    "scientific_name": [],  # handled by regex below
}

# *Genus species* or **Genus species** binomial
_SCI_RE = re.compile(r"\*{1,2}[A-Z][a-z]{2,}\s+[a-z]{3,}\*{1,2}")

_TOKEN_RE = re.compile(r"\b[a-z]{5,}\b")
_STOP = {
    "image", "images", "exhibits", "which", "these", "there", "their", "based",
    "plant", "leaf", "leaves", "provided", "visual", "appears", "showing",
    "symptoms", "specific", "characteristics", "cannot", "would", "could",
    "shown", "given", "における", "class", "belongs", "belonging", "photograph",
}


def has_marker(text: str, text_lc: str, family: str) -> bool:
    """scientific_name needs original casing; the rest match on lowercase."""
    if family == "scientific_name":
        return bool(_SCI_RE.search(text))
    return any(p in text_lc for p in MARKERS[family])


# Ground-truth labels that are catch-all / non-specific. A vague or abstaining
# answer can be judged "correct" against these, which inflates marker lifts, so
# they are analysed as a separate stratum.
_NONSPECIFIC_RE = re.compile(
    r"healthy|normal|fresh|no[_\s-]?disease|nodisease|uninfected|good", re.IGNORECASE
)
_NONSPECIFIC_EXACT = {
    "negative", "grass", "broadleaf", "soil", "background",
    "other", "unknown", "none", "nan",
}


def is_nonspecific_class(gt: str) -> bool:
    """True when the GT label is a catch-all rather than a specific taxon/disease."""
    g = (gt or "").strip().lower()
    return g in _NONSPECIFIC_EXACT or bool(_NONSPECIFIC_RE.search(g))


def fingerprint(text: str) -> frozenset[str]:
    """Content-word set used to decide whether two predictions agree."""
    toks = {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP}
    return frozenset(toks)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


AGREE_THRESHOLD = 0.5  # jaccard >= this ⇒ "same diagnosis"


# ── Step Q/A extraction ───────────────────────────────────────────────────────

def extract_step_qa(reasoning: str) -> tuple[str, str]:
    """Return (question, answer) for the LAST step in a cumulative reasoning blob."""
    blocks = re.split(r"\n\nQ:\s*", reasoning.strip())
    last = blocks[-1].strip()
    if last.startswith("Q: "):
        last = last[3:]
    elif last.startswith("Q:"):
        last = last[2:]
    for sep in ("\nA: ", "\nA:"):
        if sep in last:
            q, a = last.split(sep, 1)
            return q.strip(), a.strip()
    return "", last.strip()


# ── Marker accuracy table ─────────────────────────────────────────────────────

class MarkerTally:
    """Accumulates correct/total counts split by marker presence."""

    def __init__(self) -> None:
        self.with_n = defaultdict(int)
        self.with_c = defaultdict(int)
        self.without_n = defaultdict(int)
        self.without_c = defaultdict(int)

    def add(self, text: str, correct: int) -> None:
        text_lc = text.lower()
        for fam in MARKERS:
            if has_marker(text, text_lc, fam):
                self.with_n[fam] += 1
                self.with_c[fam] += correct
            else:
                self.without_n[fam] += 1
                self.without_c[fam] += correct

    def table(self) -> dict[str, dict]:
        out = {}
        for fam in MARKERS:
            wn, wc = self.with_n[fam], self.with_c[fam]
            on, oc = self.without_n[fam], self.without_c[fam]
            acc_w = wc / wn if wn else float("nan")
            acc_o = oc / on if on else float("nan")
            total = wn + on
            out[fam] = {
                "n_with": wn,
                "n_without": on,
                "prevalence": wn / total if total else 0.0,
                "acc_with": acc_w,
                "acc_without": acc_o,
                "lift": (acc_w - acc_o) if (wn and on) else float("nan"),
            }
        return out


# ── Loaders ───────────────────────────────────────────────────────────────────

def stance(text: str) -> str:
    """Coarse epistemic stance of an answer — used for calibration analysis."""
    tl = text.lower()
    if has_marker(text, tl, "abstention") or has_marker(text, tl, "hedge_strong"):
        return "abstains"
    if has_marker(text, tl, "hedge_soft"):
        return "hedges"
    return "commits"


def analyse_final_answers(path: Path) -> dict:
    """Marker analysis on final answers, split by condition and GT stratum."""
    return analyse_final_answers_rows(list(csv.DictReader(open(path, encoding="utf-8"))))


def analyse_final_answers_rows(rows: list[dict]) -> dict:
    """Same as analyse_final_answers but takes pre-loaded rows — lets callers
    build a synthetic final-answer set (e.g. self_cot rows from one source
    combined with cot_yaml rows derived from a different pipeline)."""
    # (cond, stratum) -> tally
    tallies: dict[tuple, MarkerTally] = defaultdict(MarkerTally)
    calib: dict[tuple, list] = defaultdict(lambda: [0, 0])  # (cond,stratum,stance)->[correct,total]
    length_bins: dict[str, list] = defaultdict(list)
    overall: dict[tuple, list] = defaultdict(lambda: [0, 0])

    for r in rows:
        jc = r.get("judged_correct", "")
        if jc == "":
            continue
        cond = r.get("condition", "")
        correct = int(jc)
        pred = r.get("prediction", "") or ""
        strat = ("nonspecific_gt" if is_nonspecific_class(r.get("ground_truth", ""))
                 else "specific_gt")

        for s in ("all", strat):
            tallies[(cond, s)].add(pred, correct)
            overall[(cond, s)][0] += correct
            overall[(cond, s)][1] += 1
            k = (cond, s, stance(pred))
            calib[k][0] += correct
            calib[k][1] += 1
        length_bins[cond].append((len(pred.split()), correct))

    conds = {c for c, _ in overall}
    result = {}
    for cond in conds:
        lens = length_bins[cond]
        wc = np.array([w for w, _ in lens])
        cc = np.array([c for _, c in lens])
        qs = np.percentile(wc, [25, 50, 75])
        buckets, edges = [], [-np.inf, *qs, np.inf]
        for i in range(4):
            m = (wc > edges[i]) & (wc <= edges[i + 1])
            if m.sum():
                lo = int(max(edges[i], 0))
                hi = int(edges[i + 1]) if np.isfinite(edges[i + 1]) else "max"
                buckets.append({"range": f"{lo}-{hi} words",
                                "n": int(m.sum()), "acc": float(cc[m].mean())})

        strata = {}
        for s in ("all", "specific_gt", "nonspecific_gt"):
            if overall[(cond, s)][1] == 0:
                continue
            strata[s] = {
                "n": overall[(cond, s)][1],
                "accuracy": overall[(cond, s)][0] / overall[(cond, s)][1],
                "markers": tallies[(cond, s)].table(),
                "calibration": {
                    st: {"n": calib[(cond, s, st)][1],
                         "acc": calib[(cond, s, st)][0] / calib[(cond, s, st)][1]}
                    for st in ("commits", "hedges", "abstains")
                    if calib[(cond, s, st)][1]
                },
            }

        result[cond] = {
            "n": overall[(cond, "all")][1],
            "accuracy": overall[(cond, "all")][0] / overall[(cond, "all")][1],
            "markers": tallies[(cond, "all")].table(),
            "length_buckets": buckets,
            "strata": strata,
        }
    return result


def analyse_steps(path: Path) -> dict:
    """Marker + trajectory + consistency analysis on cot_yaml intermediate steps."""
    step_tally: dict[int, MarkerTally] = defaultdict(MarkerTally)
    answer_tally: dict[str, MarkerTally] = defaultdict(MarkerTally)  # by stratum
    pred_tally = MarkerTally()        # markers in the step PREDICTION text
    step_acc: dict[int, list] = defaultdict(lambda: [0, 0])
    step_acc_track: dict[tuple, list] = defaultdict(lambda: [0, 0])  # (track,stratum,step)
    step_calib: dict[tuple, list] = defaultdict(lambda: [0, 0])      # (stratum,stance)
    question_text: dict[int, str] = {}

    # trajectory: (ds, image, cls) -> list of (step, correct, fingerprint)
    cumulative: dict[tuple, list] = defaultdict(list)
    independent: dict[tuple, list] = defaultdict(list)

    for r in csv.DictReader(open(path, encoding="utf-8")):
        jc = r.get("judged_correct", "")
        if jc == "":
            continue
        correct = int(jc)
        mode = r.get("mode", "")
        step = int(r.get("step_idx", 0))
        pred = r.get("prediction", "") or ""
        q, a = extract_step_qa(r.get("reasoning", "") or "")

        key = (r.get("agml_dataset", ""), r.get("image_name", ""), r.get("class_label", ""))
        strat = ("nonspecific_gt" if is_nonspecific_class(r.get("ground_truth", ""))
                 else "specific_gt")
        fp = fingerprint(pred)
        if mode == "cumulative":
            cumulative[key].append((step, correct, fp, stance(pred), strat))
            step_acc[step][0] += correct
            step_acc[step][1] += 1
            track = ("weed" if r.get("task_type") == "weed_identification"
                     else "disease_pest")
            step_acc_track[(track, strat, step)][0] += correct
            step_acc_track[(track, strat, step)][1] += 1
            step_tally[step].add(a, correct)
            pred_tally.add(pred, correct)
            for s in ("all", strat):
                answer_tally[s].add(a, correct)
                ck = (s, stance(a))
                step_calib[ck][0] += correct
                step_calib[ck][1] += 1
            if step not in question_text and q:
                question_text[step] = q
        elif mode == "independent":
            independent[key].append((step, correct, fp, "", strat))

    # ── Trajectory (cumulative) ───────────────────────────────────────────────
    traj = {
        "n_groups": 0,
        "step1_correct_final_correct": 0,
        "step1_correct_total": 0,
        "step1_wrong_final_correct": 0,
        "step1_wrong_total": 0,
        "never_changed": 0,
        "flip_counts": defaultdict(int),
        "flips_vs_final": defaultdict(lambda: [0, 0]),
    }
    for key, entries in cumulative.items():
        entries.sort(key=lambda e: e[0])
        if len(entries) < 2:
            continue
        traj["n_groups"] += 1
        first_ok, final_ok = entries[0][1], entries[-1][1]
        if first_ok:
            traj["step1_correct_total"] += 1
            traj["step1_correct_final_correct"] += final_ok
        else:
            traj["step1_wrong_total"] += 1
            traj["step1_wrong_final_correct"] += final_ok

        # Count diagnosis changes between consecutive steps
        flips = sum(
            1 for i in range(len(entries) - 1)
            if jaccard(entries[i][2], entries[i + 1][2]) < AGREE_THRESHOLD
        )
        traj["flip_counts"][flips] += 1
        traj["flips_vs_final"][flips][0] += final_ok
        traj["flips_vs_final"][flips][1] += 1
        if flips == 0:
            traj["never_changed"] += 1

    # ── Self-consistency (independent mode) ───────────────────────────────────
    consistency_buckets: dict[str, list] = defaultdict(lambda: [0, 0])
    consistency_raw: list[tuple[float, int]] = []
    for key, entries in independent.items():
        if len(entries) < 2:
            continue
        entries.sort(key=lambda e: e[0])
        fps = [e[2] for e in entries]
        sims = [
            jaccard(fps[i], fps[j])
            for i in range(len(fps)) for j in range(i + 1, len(fps))
        ]
        mean_sim = float(np.mean(sims)) if sims else 1.0
        # majority correctness across the independent probes
        maj = 1 if sum(e[1] for e in entries) * 2 > len(entries) else 0
        consistency_raw.append((mean_sim, maj))
        b = ("high (>=0.6)" if mean_sim >= 0.6
             else "medium (0.3-0.6)" if mean_sim >= 0.3
             else "low (<0.3)")
        consistency_buckets[b][0] += maj
        consistency_buckets[b][1] += 1

    sims_arr = np.array([s for s, _ in consistency_raw])
    corr_arr = np.array([c for _, c in consistency_raw])
    point_biserial = (float(np.corrcoef(sims_arr, corr_arr)[0, 1])
                      if len(sims_arr) > 2 and sims_arr.std() > 0 else float("nan"))

    # ── Per-group records for verifier-rule evaluation ────────────────────────
    sim_by_key: dict[tuple, float] = {}
    for key, entries in independent.items():
        if len(entries) < 2:
            continue
        entries.sort(key=lambda e: e[0])
        fps = [e[2] for e in entries]
        sims = [jaccard(fps[i], fps[j])
                for i in range(len(fps)) for j in range(i + 1, len(fps))]
        sim_by_key[key] = float(np.mean(sims)) if sims else 1.0

    records: list[dict] = []
    for key, entries in cumulative.items():
        if len(entries) < 2:
            continue
        entries.sort(key=lambda e: e[0])
        flips = sum(1 for i in range(len(entries) - 1)
                    if jaccard(entries[i][2], entries[i + 1][2]) < AGREE_THRESHOLD)
        records.append({
            "final_correct": entries[-1][1],
            "step1_correct": entries[0][1],
            "final_stance": entries[-1][3],
            "stratum": entries[-1][4],
            "flips": flips,
            "mean_sim": sim_by_key.get(key, float("nan")),
        })

    rules = evaluate_rules(records)

    return {
        "step_accuracy": {
            str(s): {"acc": v[0] / v[1], "n": v[1]}
            for s, v in sorted(step_acc.items())
        },
        "step_accuracy_by_track": {
            f"{t}/{s}": {
                str(st): {"acc": v[0] / v[1], "n": v[1]}
                for (t2, s2, st), v in sorted(step_acc_track.items())
                if (t2, s2) == (t, s)
            }
            for t in ("disease_pest", "weed")
            for s in ("specific_gt", "nonspecific_gt")
        },
        "step_questions": {str(k): v for k, v in sorted(question_text.items())},
        "markers_in_step_answer": answer_tally["all"].table(),
        "markers_in_step_answer_by_stratum": {
            s: t.table() for s, t in answer_tally.items()
        },
        "step_calibration": {
            f"{s}/{st}": {"n": v[1], "acc": v[0] / v[1]}
            for (s, st), v in sorted(step_calib.items()) if v[1]
        },
        "markers_in_step_prediction": pred_tally.table(),
        "markers_by_step": {
            str(s): t.table() for s, t in sorted(step_tally.items())
        },
        "trajectory": {
            "n_groups": traj["n_groups"],
            "anchor_rate": (traj["step1_correct_final_correct"] / traj["step1_correct_total"]
                            if traj["step1_correct_total"] else float("nan")),
            "recovery_rate": (traj["step1_wrong_final_correct"] / traj["step1_wrong_total"]
                              if traj["step1_wrong_total"] else float("nan")),
            "step1_correct_total": traj["step1_correct_total"],
            "step1_wrong_total": traj["step1_wrong_total"],
            "never_changed_frac": traj["never_changed"] / traj["n_groups"] if traj["n_groups"] else 0,
            "flip_distribution": {str(k): v for k, v in sorted(traj["flip_counts"].items())},
            "final_acc_by_flips": {
                str(k): {"acc": v[0] / v[1], "n": v[1]}
                for k, v in sorted(traj["flips_vs_final"].items()) if v[1] >= 50
            },
        },
        "self_consistency": {
            "buckets": {
                k: {"acc": v[0] / v[1], "n": v[1]}
                for k, v in consistency_buckets.items() if v[1]
            },
            "correlation_sim_vs_correct": point_biserial,
            "n": len(consistency_raw),
        },
        "verifier_rules": rules,
    }


def analyse_steps_swap(swap_path: Path, rotation: str = "0") -> dict:
    """Same output shape as analyse_steps(), sourced from cot_step_analysis's
    --swap_order checkpoint instead of the full-test-split step checkpoint.

    Only mode=='cumulative' rows exist in swap-order data (no independent
    mode was run), so:
      - step_accuracy / markers / trajectory / step_calibration come from the
        single `rotation` given (default "0" = natural, unrotated question
        order) — the direct analog of the original single-order run.
      - self_consistency is redefined as CROSS-ROTATION agreement: for each
        image, the final-step prediction under every rotation it was tested
        with is treated as one "probe" and compared pairwise, mirroring the
        independent-mode agreement measure structurally but built from
        genuinely different question orderings rather than genuinely
        independent (no shared context) questions. Reported under the same
        keys so downstream chart/report code needs no changes; callers should
        label this distinction explicitly in any write-up.
    """
    step_tally: dict[int, MarkerTally] = defaultdict(MarkerTally)
    answer_tally: dict[str, MarkerTally] = defaultdict(MarkerTally)
    pred_tally = MarkerTally()
    step_acc: dict[int, list] = defaultdict(lambda: [0, 0])
    step_acc_track: dict[tuple, list] = defaultdict(lambda: [0, 0])
    step_calib: dict[tuple, list] = defaultdict(lambda: [0, 0])
    question_text: dict[int, str] = {}

    cumulative: dict[tuple, list] = defaultdict(list)  # rotation-filtered trajectory
    final_by_rotation: dict[tuple, dict[str, tuple]] = defaultdict(dict)  # key -> {rot: (correct, fp, step)}

    with open(swap_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            jc = r.get("judged_correct", "")
            if jc == "" or r.get("mode", "") != "cumulative":
                continue
            correct = int(jc)
            step = int(r.get("step_idx", 0))
            rot = r.get("rotation", "")
            pred = r.get("prediction", "") or ""
            key = (r.get("agml_dataset", ""), r.get("image_name", ""), r.get("class_label", ""))
            strat = ("nonspecific_gt" if is_nonspecific_class(r.get("ground_truth", ""))
                     else "specific_gt")
            fp = fingerprint(pred)

            # Track the last-seen-so-far prediction per rotation for
            # cross-rotation agreement, independent of the primary rotation.
            prev = final_by_rotation[key].get(rot)
            if prev is None or step > prev[2]:
                final_by_rotation[key][rot] = (correct, fp, step)

            if rot != rotation:
                continue  # everything below uses only the primary rotation

            q, a = extract_step_qa(r.get("reasoning", "") or "")
            cumulative[key].append((step, correct, fp, stance(pred), strat))
            step_acc[step][0] += correct
            step_acc[step][1] += 1
            track = ("weed" if r.get("task_type") == "weed_identification"
                     else "disease_pest")
            step_acc_track[(track, strat, step)][0] += correct
            step_acc_track[(track, strat, step)][1] += 1
            step_tally[step].add(a, correct)
            pred_tally.add(pred, correct)
            for s in ("all", strat):
                answer_tally[s].add(a, correct)
                ck = (s, stance(a))
                step_calib[ck][0] += correct
                step_calib[ck][1] += 1
            if step not in question_text and q:
                question_text[step] = q

    # ── Trajectory (cumulative, primary rotation only) ────────────────────────
    traj = {
        "n_groups": 0,
        "step1_correct_final_correct": 0,
        "step1_correct_total": 0,
        "step1_wrong_final_correct": 0,
        "step1_wrong_total": 0,
        "never_changed": 0,
        "flip_counts": defaultdict(int),
        "flips_vs_final": defaultdict(lambda: [0, 0]),
    }
    for key, entries in cumulative.items():
        entries.sort(key=lambda e: e[0])
        if len(entries) < 2:
            continue
        traj["n_groups"] += 1
        first_ok, final_ok = entries[0][1], entries[-1][1]
        if first_ok:
            traj["step1_correct_total"] += 1
            traj["step1_correct_final_correct"] += final_ok
        else:
            traj["step1_wrong_total"] += 1
            traj["step1_wrong_final_correct"] += final_ok

        flips = sum(
            1 for i in range(len(entries) - 1)
            if jaccard(entries[i][2], entries[i + 1][2]) < AGREE_THRESHOLD
        )
        traj["flip_counts"][flips] += 1
        traj["flips_vs_final"][flips][0] += final_ok
        traj["flips_vs_final"][flips][1] += 1
        if flips == 0:
            traj["never_changed"] += 1

    # ── Cross-rotation agreement (substitutes independent-mode consistency) ───
    consistency_buckets: dict[str, list] = defaultdict(lambda: [0, 0])
    consistency_raw: list[tuple[float, int]] = []
    sim_by_key: dict[tuple, float] = {}
    for key, by_rot in final_by_rotation.items():
        if len(by_rot) < 2:
            continue
        fps = [v[1] for v in by_rot.values()]
        corrects = [v[0] for v in by_rot.values()]
        sims = [jaccard(fps[i], fps[j])
                for i in range(len(fps)) for j in range(i + 1, len(fps))]
        mean_sim = float(np.mean(sims)) if sims else 1.0
        sim_by_key[key] = mean_sim
        maj = 1 if sum(corrects) * 2 > len(corrects) else 0
        consistency_raw.append((mean_sim, maj))
        b = ("high (>=0.6)" if mean_sim >= 0.6
             else "medium (0.3-0.6)" if mean_sim >= 0.3
             else "low (<0.3)")
        consistency_buckets[b][0] += maj
        consistency_buckets[b][1] += 1

    sims_arr = np.array([s for s, _ in consistency_raw])
    corr_arr = np.array([c for _, c in consistency_raw])
    point_biserial = (float(np.corrcoef(sims_arr, corr_arr)[0, 1])
                      if len(sims_arr) > 2 and sims_arr.std() > 0 else float("nan"))

    # ── Per-group records for verifier-rule evaluation ────────────────────────
    records: list[dict] = []
    for key, entries in cumulative.items():
        if len(entries) < 2:
            continue
        entries.sort(key=lambda e: e[0])
        flips = sum(1 for i in range(len(entries) - 1)
                    if jaccard(entries[i][2], entries[i + 1][2]) < AGREE_THRESHOLD)
        records.append({
            "final_correct": entries[-1][1],
            "step1_correct": entries[0][1],
            "final_stance": entries[-1][3],
            "stratum": entries[-1][4],
            "flips": flips,
            "mean_sim": sim_by_key.get(key, float("nan")),
        })

    rules = evaluate_rules(records)

    return {
        "step_accuracy": {
            str(s): {"acc": v[0] / v[1], "n": v[1]}
            for s, v in sorted(step_acc.items())
        },
        "step_accuracy_by_track": {
            f"{t}/{s}": {
                str(st): {"acc": v[0] / v[1], "n": v[1]}
                for (t2, s2, st), v in sorted(step_acc_track.items())
                if (t2, s2) == (t, s)
            }
            for t in ("disease_pest", "weed")
            for s in ("specific_gt", "nonspecific_gt")
        },
        "step_questions": {str(k): v for k, v in sorted(question_text.items())},
        "markers_in_step_answer": answer_tally["all"].table(),
        "markers_in_step_answer_by_stratum": {
            s: t.table() for s, t in answer_tally.items()
        },
        "step_calibration": {
            f"{s}/{st}": {"n": v[1], "acc": v[0] / v[1]}
            for (s, st), v in sorted(step_calib.items()) if v[1]
        },
        "markers_in_step_prediction": pred_tally.table(),
        "markers_by_step": {
            str(s): t.table() for s, t in sorted(step_tally.items())
        },
        "trajectory": {
            "n_groups": traj["n_groups"],
            "anchor_rate": (traj["step1_correct_final_correct"] / traj["step1_correct_total"]
                            if traj["step1_correct_total"] else float("nan")),
            "recovery_rate": (traj["step1_wrong_final_correct"] / traj["step1_wrong_total"]
                              if traj["step1_wrong_total"] else float("nan")),
            "step1_correct_total": traj["step1_correct_total"],
            "step1_wrong_total": traj["step1_wrong_total"],
            "never_changed_frac": traj["never_changed"] / traj["n_groups"] if traj["n_groups"] else 0,
            "flip_distribution": {str(k): v for k, v in sorted(traj["flip_counts"].items())},
            "final_acc_by_flips": {
                str(k): {"acc": v[0] / v[1], "n": v[1]}
                for k, v in sorted(traj["flips_vs_final"].items()) if v[1] >= 20
            },
        },
        "self_consistency": {
            "buckets": {
                k: {"acc": v[0] / v[1], "n": v[1]}
                for k, v in consistency_buckets.items() if v[1]
            },
            "correlation_sim_vs_correct": point_biserial,
            "n": len(consistency_raw),
            "method": "cross_rotation_agreement",
        },
        "verifier_rules": rules,
    }


def analyse_final_answers_swap(cot_vision_path: Path, swap_path: Path,
                               rotation: str = "0") -> dict:
    """final_answers-shaped dict combining self_cot rows read directly from the
    (still-valid, prompt-change-independent) cot_vision final-answer CSV with
    synthetic cot_yaml rows built from the swap-order checkpoint — the last
    chain step per image under the given rotation, i.e. exactly the model's
    graded final answer at that point.

    self_cot is restricted to the EXACT SAME images as the cot_yaml sample
    (matched on dataset + image basename + class label, not just dataset) so
    the two conditions' accuracy numbers are a same-sample, apples-to-apples
    comparison rather than two differently-sized pools over the same datasets.
    """
    def img_key(dataset: str, image_name: str, class_label: str) -> tuple:
        return (dataset, Path(image_name).name, class_label)

    by_key: dict[tuple, dict] = {}
    with open(swap_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("mode") != "cumulative" or r.get("rotation") != rotation:
                continue
            if r.get("judged_correct", "") == "":
                continue
            key = img_key(r["agml_dataset"], r["image_name"], r["class_label"])
            step = int(r["step_idx"])
            prev = by_key.get(key)
            if prev is None or step > prev["_step"]:
                by_key[key] = {
                    "condition": "cot_yaml",
                    "prediction": r.get("prediction", ""),
                    "ground_truth": r.get("ground_truth", ""),
                    "judged_correct": r.get("judged_correct", ""),
                    "_step": step,
                }

    swap_keys = set(by_key.keys())
    self_cot_by_key: dict[tuple, dict] = {}
    for r in csv.DictReader(open(cot_vision_path, encoding="utf-8")):
        if r.get("condition") != "self_cot":
            continue
        k = img_key(r.get("agml_dataset", ""), r.get("image_name", ""), r.get("class_label", ""))
        if k in swap_keys:
            self_cot_by_key[k] = r

    # Intersect both ways: a handful of swap-sample images (~2%) have no
    # self_cot counterpart (different underlying test-split sampling), so
    # drop those from cot_yaml too — n must be identical, not just close.
    common_keys = swap_keys & set(self_cot_by_key.keys())
    cot_yaml_rows = [by_key[k] for k in common_keys]
    self_cot_rows = [self_cot_by_key[k] for k in common_keys]

    return analyse_final_answers_rows(self_cot_rows + cot_yaml_rows)


# ── Verifier rule evaluation ──────────────────────────────────────────────────

RULES: dict[str, str] = {
    "abstains":              "final answer abstains / hedges strongly",
    "low_consistency":       "cross-probe agreement < 0.3",
    "many_flips":            "diagnosis changed >= 4 times across steps",
    "abstains_OR_lowcons":   "abstains OR agreement < 0.3",
    "lowcons_AND_manyflips": "agreement < 0.3 AND >= 4 flips",
    "any_of_three":          "abstains OR agreement < 0.3 OR >= 4 flips",
}


def _rule_fires(rec: dict, rule: str) -> bool:
    abst = rec["final_stance"] == "abstains"
    low = rec["mean_sim"] < 0.3
    many = rec["flips"] >= 4
    return {
        "abstains": abst,
        "low_consistency": low,
        "many_flips": many,
        "abstains_OR_lowcons": abst or low,
        "lowcons_AND_manyflips": low and many,
        "any_of_three": abst or low or many,
    }[rule]


def evaluate_rules(records: list[dict]) -> dict:
    """For each candidate flag rule report coverage, precision, recall and the
    accuracy of the answers left UNflagged (i.e. what you'd auto-accept)."""
    out: dict[str, dict] = {}
    for stratum in ("all", "specific_gt"):
        recs = [r for r in records
                if stratum == "all" or r["stratum"] == stratum]
        if not recs:
            continue
        n = len(recs)
        n_wrong = sum(1 for r in recs if not r["final_correct"])
        base_acc = sum(r["final_correct"] for r in recs) / n

        stratum_out = {"n": n, "base_accuracy": base_acc,
                       "wrong_rate": n_wrong / n, "rules": {}}
        for rule, desc in RULES.items():
            flagged = [r for r in recs if _rule_fires(r, rule)]
            kept = [r for r in recs if not _rule_fires(r, rule)]
            nf = len(flagged)
            if nf == 0 or not kept:
                continue
            flagged_wrong = sum(1 for r in flagged if not r["final_correct"])
            stratum_out["rules"][rule] = {
                "description": desc,
                "coverage": nf / n,
                "precision": flagged_wrong / nf,      # flagged items that really are wrong
                "recall": flagged_wrong / n_wrong if n_wrong else float("nan"),
                "kept_n": len(kept),
                "kept_accuracy": sum(r["final_correct"] for r in kept) / len(kept),
                "kept_acc_gain": sum(r["final_correct"] for r in kept) / len(kept) - base_acc,
            }
        out[stratum] = stratum_out
    return out


# ── Charts ────────────────────────────────────────────────────────────────────

def make_charts(results: dict, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []

    # 1. Marker lift, final answers, both conditions
    # Not sharey: the two conditions can have different numbers of marker
    # categories (e.g. one marker may never fire for one condition), and a
    # shared y-scale previously crammed the taller panel's top label against
    # the subplot title, clipping it invisibly.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, cond in zip(axes, ["self_cot", "cot_yaml"]):
        m = (results["final_answers"].get(cond, {})
             .get("strata", {}).get("specific_gt", {}).get("markers", {}))
        fams = [f for f in m if not np.isnan(m[f]["lift"])]
        fams.sort(key=lambda f: m[f]["lift"])
        lifts = [m[f]["lift"] * 100 for f in fams]
        colors = ["#c62828" if v < 0 else "#2e7d32" for v in lifts]
        ax.barh(range(len(fams)), lifts, color=colors, alpha=0.85)
        ax.set_yticks(range(len(fams)))
        ax.set_yticklabels([f.replace("_", " ") for f in fams], fontsize=9)
        ax.set_ylim(-0.6, len(fams) - 0.3)  # padding so the top label clears the title
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Accuracy lift when marker present (pp)", fontsize=9)
        ax.set_title(f"{cond} — final answer markers (specific GT only)",
                     fontsize=11, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.25)
        # Anchor every label to the right of its bar end (or of zero for negative
        # bars) so nothing collides with the y-axis tick labels.
        for i, (f, v) in enumerate(zip(fams, lifts)):
            ax.annotate(f"{v:+.1f}  (n={m[f]['n_with']:,})",
                        (max(v, 0), i), fontsize=7, va="center", ha="left",
                        xytext=(4, 0), textcoords="offset points")
        lo, hi = min(lifts + [0]), max(lifts + [0])
        ax.set_xlim(lo - abs(lo) * 0.10 - 1, hi + max(abs(hi), 8) * 1.25)
    plt.tight_layout()
    p = out_dir / "marker_lift_final.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # 2. Step accuracy + marker prevalence by step
    steps_d = results["steps"]["step_accuracy"]
    steps = sorted(int(s) for s in steps_d)
    accs = [steps_d[str(s)]["acc"] * 100 for s in steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(steps, accs, marker="o", color="#1f77b4", linewidth=2)
    for s, a in zip(steps, accs):
        ax1.annotate(f"{a:.1f}%", (s, a), fontsize=8, ha="center",
                     xytext=(0, 7), textcoords="offset points")
    ax1.set_xticks(steps); ax1.set_xlabel("CoT step", fontsize=10)
    ax1.set_ylabel("Judged accuracy (%)", fontsize=10)
    ax1.set_title("Accuracy after each cumulative step", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.25)

    by_step = results["steps"]["markers_by_step"]
    for fam, color in [("abstention", "#c62828"), ("image_quality_complaint", "#ef6c00"),
                       ("scientific_name", "#2e7d32"), ("disclaimer", "#6a1b9a")]:
        ys = [by_step[str(s)][fam]["prevalence"] * 100 for s in steps if str(s) in by_step]
        ax2.plot(steps[:len(ys)], ys, marker="o", label=fam.replace("_", " "), color=color)
    ax2.set_xticks(steps); ax2.set_xlabel("CoT step", fontsize=10)
    ax2.set_ylabel("Prevalence in step answer (%)", fontsize=10)
    ax2.set_title("Marker prevalence across steps", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25)
    plt.tight_layout()
    p = out_dir / "step_accuracy_markers.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # 2b. Stratified step trajectory — the headline chart
    bt = results["steps"]["step_accuracy_by_track"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    style = {
        "disease_pest/specific_gt":    ("#2e7d32", "-",  "Disease/pest — real disease GT"),
        "disease_pest/nonspecific_gt": ("#c62828", "--", "Disease/pest — healthy GT"),
        "weed/specific_gt":            ("#1565c0", "-",  "Weed — named species GT"),
        "weed/nonspecific_gt":         ("#ef6c00", "--", "Weed — coarse/negative GT"),
    }
    for key, (color, ls, label) in style.items():
        d = bt.get(key, {})
        if not d:
            continue
        xs = sorted(int(k) for k in d)
        ys = [d[str(x)]["acc"] * 100 for x in xs]
        ax.plot(xs, ys, marker="o", color=color, linestyle=ls, linewidth=2, label=label)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}", (x, y), fontsize=7, ha="center",
                        xytext=(0, 6), textcoords="offset points", color=color)
    ax.set_xlabel("CoT step", fontsize=10)
    ax.set_ylabel("Judged accuracy (%)", fontsize=10)
    ax.set_title("CoT helps on real disease, collapses on healthy tissue",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8.5, loc="center right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    p = out_dir / "step_accuracy_stratified.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # 3. Trajectory + self-consistency
    traj = results["steps"]["trajectory"]
    cons = results["steps"]["self_consistency"]["buckets"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fbf = traj["final_acc_by_flips"]
    ks = sorted(int(k) for k in fbf)
    ax1.bar([str(k) for k in ks], [fbf[str(k)]["acc"] * 100 for k in ks],
            color="#1f77b4", alpha=0.85)
    for i, k in enumerate(ks):
        ax1.annotate(f"{fbf[str(k)]['acc']*100:.1f}%\nn={fbf[str(k)]['n']:,}",
                     (i, fbf[str(k)]["acc"] * 100), fontsize=8, ha="center",
                     xytext=(0, 4), textcoords="offset points")
    ax1.set_xlabel("Number of diagnosis changes across steps", fontsize=10)
    ax1.set_ylabel("Final judged accuracy (%)", fontsize=10)
    ax1.set_title("Instability vs final correctness (cumulative)", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.25)

    order = ["low (<0.3)", "medium (0.3-0.6)", "high (>=0.6)"]
    order = [o for o in order if o in cons]
    ax2.bar(order, [cons[o]["acc"] * 100 for o in order], color="#ff7f0e", alpha=0.85)
    for i, o in enumerate(order):
        ax2.annotate(f"{cons[o]['acc']*100:.1f}%\nn={cons[o]['n']:,}",
                     (i, cons[o]["acc"] * 100), fontsize=8, ha="center",
                     xytext=(0, 4), textcoords="offset points")
    cons_method = results["steps"]["self_consistency"].get("method", "independent_mode")
    cons_label = ("Cross-rotation agreement (swap-order)" if cons_method == "cross_rotation_agreement"
                  else "Cross-probe agreement (independent mode)")
    ax2.set_xlabel(cons_label, fontsize=10)
    ax2.set_ylabel("Majority-vote accuracy (%)", fontsize=10)
    ax2.set_title("Self-consistency vs correctness", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    p = out_dir / "trajectory_consistency.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    return paths


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--cot_dir", default="outputs/cot_vision/")
    ap.add_argument("--step_dir", default="outputs/cot_step_analysis/")
    ap.add_argument("--output_dir", default="outputs/verifier_analysis/")
    ap.add_argument("--source", choices=["full", "swap"], default="full",
                    help="'full' = original full-test-split ckpt_judged.csv; "
                         "'swap' = the --swap_order subsample (10 datasets, "
                         "<=50 img/class), using rotation 0 as the primary "
                         "(unrotated) order. Use 'swap' to refresh the "
                         "analysis quickly against the current prompts "
                         "without a full cot_step_analysis rerun.")
    ap.add_argument("--rotation", default="0",
                    help="Primary rotation to use as the natural question "
                         "order, only relevant with --source swap.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe = args.model.replace("/", "_")
    final_path = Path(args.cot_dir) / f"cot_cls_{safe}.csv"

    if args.source == "swap":
        step_path = Path(args.step_dir) / "swap_order" / "ckpt_swap_judged.csv"
        json_path = out_dir / "verifier_analysis_swap.json"

        print(f"Analysing final answers (self_cot from {final_path.name} + "
              f"cot_yaml derived from swap rotation {args.rotation}): {step_path}")
        final_res = analyse_final_answers_swap(final_path, step_path, rotation=args.rotation)
        for cond, d in final_res.items():
            print(f"  {cond}: n={d['n']:,} acc={d['accuracy']:.3f}")

        print(f"\nAnalysing intermediate steps (swap subsample, rotation {args.rotation}): {step_path}")
        step_res = analyse_steps_swap(step_path, rotation=args.rotation)
    else:
        step_path = Path(args.step_dir) / "ckpt_judged.csv"
        json_path = out_dir / "verifier_analysis.json"

        print(f"Analysing final answers: {final_path}")
        final_res = analyse_final_answers(final_path)
        for cond, d in final_res.items():
            print(f"  {cond}: n={d['n']:,} acc={d['accuracy']:.3f}")

        print(f"\nAnalysing intermediate steps: {step_path}")
        step_res = analyse_steps(step_path)

    print(f"  groups={step_res['trajectory']['n_groups']:,}  "
          f"anchor={step_res['trajectory']['anchor_rate']:.3f}  "
          f"recovery={step_res['trajectory']['recovery_rate']:.3f}")

    results = {"final_answers": final_res, "steps": step_res, "model": args.model,
              "source": args.source}

    json_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved: {json_path}")

    chart_dir = out_dir / "swap_subsample" if args.source == "swap" else out_dir
    chart_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating charts in {chart_dir}...")
    for p in make_charts(results, chart_dir):
        print(f"  Saved: {p}")


if __name__ == "__main__":
    main()
