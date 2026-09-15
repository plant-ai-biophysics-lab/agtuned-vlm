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

_sem_sim_model = None

_SEM_SIM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

def _get_sem_sim_model():
    global _sem_sim_model
    if _sem_sim_model is None:
        from sentence_transformers import SentenceTransformer
        try:
            _sem_sim_model = SentenceTransformer(_SEM_SIM_MODEL_ID, local_files_only=True)
        except Exception:
            _sem_sim_model = SentenceTransformer(_SEM_SIM_MODEL_ID)
    return _sem_sim_model


def sem_sim_score(prediction: str, reference: str) -> float:
    """Cosine similarity between sentence embeddings of prediction and reference."""
    if not prediction or not reference:
        return 0.0
    model = _get_sem_sim_model()
    embs = model.encode([prediction, reference], convert_to_numpy=True, normalize_embeddings=True)
    return round(float(np.dot(embs[0], embs[1])), 4)
try:
    from vllm import LLM, SamplingParams
    _JUDGE_PARAMS = SamplingParams(temperature=0.0, max_tokens=1024)
except ImportError:
    LLM = None  # type: ignore[assignment,misc]
    SamplingParams = None  # type: ignore[assignment,misc]
    _JUDGE_PARAMS = None

for _corpus in ("wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{_corpus}")
    except LookupError:
        nltk.download(_corpus, quiet=True)

THINK_RE = re.compile(r"<(?:think|thought)>.*?</(?:think|thought)>", re.DOTALL | re.IGNORECASE)
_THINK_RE_JUDGE = THINK_RE  # alias kept for internal use


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


def rouge_scores_per_sample(prediction: str, reference: str, include_rouge: bool = True) -> dict[str, float]:
    """Returns {rouge1, rouge2, rougeL, rouge_ss, sem_sim} for a single
    prediction/reference pair. include_rouge=False skips the ROUGE family
    (rouge1/rouge2/rougeL's RougeScorer + rouge_ss's WordNet synonym
    expansion) and returns "" for those four — used by the classification
    scripts (vision_baseline.py, desc_augmented_vision_v2.py, self_cot.py,
    cot_yaml.py), where ROUGE was only ever a secondary partial-credit
    signal alongside the real metrics (F1 / judge_f1_macro) and isn't used.
    language_assessment/assess.py and vision_assessment/assess.py (open-ended
    QA, not classification) still call this with the default — there ROUGE-SS
    is a primary metric, not a bonus number. sem_sim is unaffected either
    way — aggregate_rouge() already skips "" values when averaging, so this
    doesn't require any downstream changes."""
    if not include_rouge:
        return {
            "rouge1": "", "rouge2": "", "rougeL": "", "rouge_ss": "",
            "sem_sim": sem_sim_score(prediction, reference),
        }
    scorer = rs_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(reference, prediction)
    return {
        "rouge1": round(s["rouge1"].fmeasure, 4),
        "rouge2": round(s["rouge2"].fmeasure, 4),
        "rougeL": round(s["rougeL"].fmeasure, 4),
        "rouge_ss": rouge_ss_score(prediction, reference),
        "sem_sim": sem_sim_score(prediction, reference),
    }


def aggregate_rouge(rows: list[dict]) -> dict[str, float]:
    """Mean of rouge1/rouge2/rougeL/rouge_ss/sem_sim over rows that have those keys."""
    keys = ["rouge1", "rouge2", "rougeL", "rouge_ss", "sem_sim"]
    result = {}
    for k in keys:
        vals = []
        for r in rows:
            if k in r and r[k] not in ("", None):
                try:
                    vals.append(float(r[k]))
                except (TypeError, ValueError):
                    pass
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


_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer_tag(text: str) -> tuple[str, bool]:
    """Pull the free-form <answer>...</answer> content out of a generated
    response — open-ended, no closed-set vocabulary (real-world answers
    aren't a fixed label set). Falls back to the full cleaned text if the
    tag is missing, so every response stays comparable rather than empty;
    `found` records which case happened. Moved here from verifier_cot.py
    (originally written for its rubric templates) so plain-classification
    stages (vision_baseline.py, self_cot.py, cot_yaml.py,
    desc_augmented_vision_v2.py) can reuse the exact same convention —
    call resolve_class on the extracted tag content instead of the raw
    free-text prediction, which is what actually determines
    resolved_class/correct in each of those scripts."""
    m = _ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).strip(), True
    return text.strip(), False


def compute_classification_metrics(refs: list[str], preds: list[str]) -> dict:
    """Compute accuracy, contains_accuracy, f1_macro, f1_weighted, n_total, n_resolvable."""
    labels = sorted(set(refs) | set(p for p in preds if p))
    # An unresolved prediction (p == "") must never count as a contains-match —
    # contains_match("", r) is True for ANY r (an empty string is a substring
    # of everything), which previously inflated contains_accuracy toward 100%
    # whenever resolve_class failed on most rows (e.g. free-text CoT/verifier
    # answers), silently masking a parsing failure as a near-perfect score.
    contains_correct = [int(bool(p) and contains_match(p, r)) for p, r in zip(preds, refs)]
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


def sample_class_images(image_names: list[str], n: int | None, seed: int = 42) -> list[str]:
    """Return up to n image names from image_names, shuffled with a fixed seed.

    Ensures vision assessment always evaluates the same images regardless of
    filesystem ordering, making results reproducible and comparable across runs.
    """
    if not n or n >= len(image_names):
        return image_names
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(image_names), size=n, replace=False)
    indices.sort()
    return [image_names[i] for i in indices]


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
    judged = [r for r in rows if "judged_correct" in r and r["judged_correct"] != ""]
    if not judged:
        return {}
    jc = [int(r["judged_correct"]) for r in judged]
    result = {
        "judge_accuracy": round(sum(jc) / len(jc), 4),
        "n_judged": len(judged),
    }
    if "resolved_class" in judged[0] and "class_label" in judged[0]:
        refs = [r["class_label"] for r in judged]
        preds = [r["class_label"] if jc[i] else r["resolved_class"] for i, r in enumerate(judged)]
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
    log_path=None,
    temperature: float | None = None,
    seed: int | None = None,
    extra_keys: dict[str, str] | None = None,
    enable_thinking: bool = False,
    max_tokens: int | None = None,
) -> list[dict]:
    """Apply LLM judge to rows; adds judged_correct (0/1) and judge_reasoning to each row.

    log_path (optional, a Path): append one JSONL record per row with the
    FULL judge conversation — system prompt, rendered chat-templated prompt,
    untruncated raw model response — plus the row's identifying fields and
    exactly what gt/pred text it was shown. This is for QA auditing what the
    judge actually saw and said; the row's own judge_reasoning field is
    truncated to 600 chars and doesn't include the prompt at all. Default
    None (every pre-existing caller) skips logging — purely additive, no
    behavior change for callers that don't pass it.

    temperature/seed (both default None): override _JUDGE_PARAMS' fixed
    temperature=0.0 for THIS call only — every existing caller leaves these
    None and keeps the exact original deterministic behavior. Added for
    src/common/judge_consistency.py's self-consistency sweep (Phase 2 of the
    judge-improvements evaluation plan), which needs repeat judge calls over
    the SAME rows to actually produce different verdicts sometimes — at
    temperature=0.0 (greedy) every repeat call is bit-for-bit identical, so
    there's nothing to measure a flip rate over.

    extra_keys (default None): {template_placeholder_name: row_field_name}
    — judge_template.format() only ever filled {gt}/{pred} before this; some
    Phase 3 template variants (src/common/judge_template_sensitivity.py)
    need a third placeholder, e.g. {class_list}, whose value varies per row
    (each dataset's own class list) rather than being a fixed string bakeable
    into the template itself. Every existing caller leaves this None and
    gets the exact original gt/pred-only formatting.

    enable_thinking (default False): every existing caller leaves this False
    and gets the EXACT original prompt-rendering behavior — plain
    tokenizer.apply_chat_template(msgs, tokenize=False,
    add_generation_prompt=True), no enable_thinking kwarg passed at all. Only
    when explicitly True does _fmt attempt
    apply_chat_template(..., enable_thinking=True), falling back to the
    plain call (via TypeError) if the tokenizer's chat template doesn't
    accept that kwarg (e.g. non-Qwen3 tokenizers). This exists to test
    whether the judge model exposes a controllable reasoning/thinking mode
    (see src/common/chat_prompt.py::render_chat_template for the same
    enable_thinking pattern used elsewhere in the pipeline) — unverified for
    Gemma-4, which may use a different channel-based reasoning format
    entirely; this flag is a no-op if the tokenizer doesn't support the
    kwarg.

    max_tokens (default None): override _JUDGE_PARAMS' fixed max_tokens=1024
    for THIS call only — every existing caller leaves this None and keeps
    the original budget. Added alongside enable_thinking since a thinking
    trace can consume (or exceed) the judge's normal 1024-token budget
    before it ever reaches its final JSON verdict, truncating the response
    mid-thought.
    """
    judge_params = _JUDGE_PARAMS if temperature is None and seed is None and max_tokens is None else SamplingParams(
        temperature=_JUDGE_PARAMS.temperature if temperature is None else temperature,
        max_tokens=_JUDGE_PARAMS.max_tokens if max_tokens is None else max_tokens,
        seed=seed,
    )

    def _fmt(text: str) -> str:
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
        if enable_thinking:
            try:
                return tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                )
            except TypeError:
                pass  # tokenizer's chat template doesn't accept enable_thinking — fall through
            except Exception:
                pass
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            msgs = [{"role": "user", "content": f"{system_prompt}\n\n{text}"}]
            try:
                return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                return text

    def _extra_fmt_kwargs(r: dict) -> dict:
        if not extra_keys:
            return {}
        return {placeholder: str(r.get(row_field, "")) for placeholder, row_field in extra_keys.items()}

    raw_prompts = [
        judge_template.format(
            gt=str(r.get(gt_key, ""))[:400], pred=str(r.get(pred_key, ""))[:400],
            **_extra_fmt_kwargs(r),
        )
        for r in rows
    ]
    formatted_prompts = [_fmt(p) for p in raw_prompts]
    outputs = llm.generate(formatted_prompts, sampling_params=judge_params)

    updated = []
    log_records = []
    for row, raw_prompt, formatted_prompt, out in zip(rows, raw_prompts, formatted_prompts, outputs):
        raw = out.outputs[0].text
        cleaned = _THINK_RE_JUDGE.sub("", raw).strip()
        judged_correct = _parse_judge_response(raw, threshold=confidence_threshold)
        updated.append({
            **row,
            "judged_correct": judged_correct,
            "judge_reasoning": cleaned[:600],
            # Completely untouched judge output, no <think>-stripping and no
            # 600-char truncation — added alongside enable_thinking for the
            # same reason vision_baseline.py's raw_response column was: a
            # thinking model's reasoning isn't guaranteed to show up as a
            # tag _THINK_RE_JUDGE would ever match, so judge_reasoning alone
            # can silently still contain it (or drop it) with no way to
            # tell. Additive key — every existing caller/row shape is
            # unaffected, this just rides along in **row for whatever
            # downstream code chooses to keep it.
            "judge_raw_response": raw,
        })
        if log_path is not None:
            log_records.append({
                "agml_dataset": row.get("agml_dataset", ""),
                "class_label": row.get("class_label", ""),
                "image_name": row.get("image_name", ""),
                "condition": row.get("condition", ""),
                "gt_key": gt_key,
                "pred_key": pred_key,
                "gt_full": str(row.get(gt_key, "")),
                "pred_full": str(row.get(pred_key, "")),
                "confidence_threshold": confidence_threshold,
                "system_prompt": system_prompt,
                "judge_user_message": raw_prompt,
                "rendered_prompt": formatted_prompt,
                "raw_response": raw,
                "judged_correct": judged_correct,
            })

    if log_path is not None and log_records:
        from src.common.io import append_jsonl
        log_path.parent.mkdir(parents=True, exist_ok=True)
        for rec in log_records:
            append_jsonl(log_path, rec)

    return updated
