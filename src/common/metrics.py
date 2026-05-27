"""Shared metric functions — used by all assessment stages."""
from __future__ import annotations
import json
import re
from collections import defaultdict
from functools import lru_cache
import hashlib

import nltk
import numpy as np
from nltk.corpus import wordnet
from rouge_score import rouge_scorer as rs_scorer
from sklearn.metrics import accuracy_score, f1_score
from vllm import LLM, SamplingParams

for _corpus in ("wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{_corpus}")
    except LookupError:
        nltk.download(_corpus, quiet=True)

_THINK_RE_JUDGE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JUDGE_PARAMS = SamplingParams(temperature=0.0, max_tokens=1024)


@lru_cache(maxsize=16384)
def _synonyms(word: str) -> frozenset[str]:
    """Return word + all WordNet lemma names as a cached frozenset."""
    syns = {word}
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            syns.add(lemma.name().replace("_", " ").lower())
    return frozenset(syns)


def rouge_ss_score(prediction: str, reference: str) -> float:
    """ROUGE-SS: unigram F1 with WordNet synonym expansion on both sides."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_expanded = set().union(*[_synonyms(t) for t in ref_tokens])
    pred_expanded = set().union(*[_synonyms(t) for t in pred_tokens])

    precision = sum(1 for t in pred_tokens if t in ref_expanded) / len(pred_tokens)
    recall = sum(1 for t in ref_tokens if t in pred_expanded) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def rouge_scores_per_sample(prediction: str, reference: str) -> dict[str, float]:
    """Returns {rouge1, rouge2, rougeL, rouge_ss} for a single prediction/reference pair."""
    scorer = rs_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(reference, prediction)
    return {
        "rouge1": round(s["rouge1"].fmeasure, 4),
        "rouge2": round(s["rouge2"].fmeasure, 4),
        "rougeL": round(s["rougeL"].fmeasure, 4),
        "rouge_ss": rouge_ss_score(prediction, reference),
    }


def aggregate_rouge(rows: list[dict]) -> dict[str, float]:
    """Mean of rouge1/rouge2/rougeL/rouge_ss over rows that have those keys."""
    keys = ["rouge1", "rouge2", "rougeL", "rouge_ss"]
    result = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            result[k] = round(float(np.mean(vals)), 4)
    return result


def naive_rouge_baseline(rows: list[dict]) -> dict[str, float]:
    """Shuffle ground_truth across rows and compute ROUGE — sets the chance floor."""
    if not rows:
        return {}
    refs = [r["ground_truth"] for r in rows]
    rng = np.random.default_rng(42)
    shuffled = refs.copy()
    rng.shuffle(shuffled)
    per_row = [rouge_scores_per_sample(s, r) for s, r in zip(shuffled, refs)]
    return aggregate_rouge(per_row)


def contains_match(pred: str, ref: str) -> bool:
    """True if either string is a substring of the other (case-insensitive)."""
    p, r = pred.lower().strip(), ref.lower().strip()
    return p in r or r in p


def resolve_class(response: str, classes: list[str]) -> str | None:
    """Case-insensitive exact match, then single-match substring fallback."""
    resp = response.lower().strip()
    for cls in classes:
        if cls.lower() == resp:
            return cls
    matches = [cls for cls in classes if cls.lower() in resp]
    return matches[0] if len(matches) == 1 else None


def compute_classification_metrics(refs: list[str], preds: list[str]) -> dict:
    """Compute accuracy, contains_accuracy, f1_macro, f1_weighted, n_total, n_resolvable."""
    labels = sorted(set(refs) | set(p for p in preds if p))
    contains_correct = [int(contains_match(p, r)) for p, r in zip(preds, refs)]
    n_resolvable = sum(1 for p in preds if p)
    return {
        "accuracy": round(accuracy_score(refs, preds), 4),
        "contains_accuracy": round(float(np.mean(contains_correct)), 4),
        "f1_macro": round(f1_score(refs, preds, average="macro", labels=labels, zero_division=0), 4),
        "f1_weighted": round(f1_score(refs, preds, average="weighted", labels=labels, zero_division=0), 4),
        "n_total": len(refs),
        "n_resolvable": n_resolvable,
    }


def naive_classification_baseline(refs: list[str]) -> dict:
    """Shuffle ground truth labels — chance-level accuracy/F1."""
    shuffled = refs.copy()
    np.random.default_rng(42).shuffle(shuffled)
    return compute_classification_metrics(refs, shuffled)


def model_color_map(models: list[str]) -> dict[str, tuple]:
    """
    Assign a consistent tab10 color to each model based on its name hash,
    so colors stay the same regardless of evaluation order or which assessment
    is producing the chart.
    """
    import matplotlib.pyplot as plt
    palette = plt.cm.tab10.colors  # 10 distinct colors
    return {m: palette[int(hashlib.md5(m.encode()).hexdigest(), 16) % len(palette)] for m in models}


def naive_uniform_baseline_per_dataset(rows: list[dict]) -> dict:
    """
    Per-dataset uniform random baseline: expected F1 = 1/k where k = number of
    unique classes in that dataset. Returns metrics macro-averaged across datasets.
    """
    by_ds: dict[str, set] = defaultdict(set)
    for row in rows:
        by_ds[row["agml_dataset"]].add(row.get("class_label", row.get("ground_truth", "")))
    per_ds_f1 = [1.0 / max(len(classes), 1) for classes in by_ds.values()]
    mean_f1 = float(np.mean(per_ds_f1)) if per_ds_f1 else 0.0
    return {"f1_macro": round(mean_f1, 4), "accuracy": round(mean_f1, 4)}


def aggregate_by_task(
    per_dataset: dict[str, dict],
    ds_to_task: dict[str, str],
) -> dict[str, dict]:
    """Group per_dataset entries by task_type, then mean each numeric metric."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for ds_name, metrics in per_dataset.items():
        tt = ds_to_task.get(ds_name)
        if tt:
            by_task[tt].append(metrics)

    result = {}
    for tt, metric_list in by_task.items():
        agg = {}
        for key in metric_list[0]:
            vals = [m[key] for m in metric_list if isinstance(m.get(key), (int, float))]
            if vals:
                agg[key] = round(float(np.mean(vals)), 4)
        result[tt] = agg
    return result


def judged_summary(rows: list[dict]) -> dict:
    """Returns judge_accuracy, n_judged, and optionally judge_f1_macro."""
    judged = [r for r in rows if "judged_correct" in r]
    if not judged:
        return {}
    result = {
        "judge_accuracy": round(sum(r["judged_correct"] for r in judged) / len(judged), 4),
        "n_judged": len(judged),
    }
    if "resolved_class" in judged[0] and "class_label" in judged[0]:
        refs = [r["class_label"] for r in judged]
        preds = [r["class_label"] if r["judged_correct"] else r["resolved_class"] for r in judged]
        labels = sorted(set(refs))
        result["judge_f1_macro"] = round(
            f1_score(refs, preds, average="macro", labels=labels, zero_division=0), 4
        )
    return result


def _parse_judge_response(text: str, threshold: int = 1) -> int:
    """Extract is_match + confidence from JSON judge response. Returns 0 or 1."""
    text = _THINK_RE_JUDGE.sub("", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            data = json.loads(text[start : end + 1])
            is_match = data.get("is_match", False)
            confidence = int(data.get("confidence", 0))
            if isinstance(is_match, str):
                is_match = is_match.lower() in ("true", "yes")
            return int(bool(is_match) and confidence >= threshold)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    for pattern in [r'"is_match"\s*:\s*(true|false)', r'"match"\s*:\s*(true|false)']:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return int(m.group(1).lower() == "true")
    return 0


def apply_judge(
    rows: list[dict],
    llm: LLM,
    tokenizer,
    system_prompt: str,
    judge_template: str,
    gt_key: str,
    pred_key: str,
    confidence_threshold: int = 1,
) -> list[dict]:
    """Apply LLM judge to rows; adds judged_correct (0/1) and judge_reasoning to each row."""

    def _fmt(text: str) -> str:
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            msgs = [{"role": "user", "content": f"{system_prompt}\n\n{text}"}]
            try:
                return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                return text

    raw_prompts = [
        judge_template.format(gt=str(r.get(gt_key, ""))[:400], pred=str(r.get(pred_key, ""))[:400])
        for r in rows
    ]
    outputs = llm.generate([_fmt(p) for p in raw_prompts], sampling_params=_JUDGE_PARAMS)

    updated = []
    for row, out in zip(rows, outputs):
        raw = out.outputs[0].text
        cleaned = _THINK_RE_JUDGE.sub("", raw).strip()
        updated.append({
            **row,
            "judged_correct": _parse_judge_response(raw, threshold=confidence_threshold),
            "judge_reasoning": cleaned[:600],
        })
    return updated
