"""
Grep/keyword-based retrieval grounding against the GARDIAN passages corpus
(experiments/ag_finetune/outputs/stage3_gardian_passages.jsonl) for
experiments/verifier_cot/run_verifier_cot.py's --database flag.

Purpose: cheap, no-training test of whether grounding the verifier's
judgment in real literature passages (instead of relying purely on the
model's own training-memory "knowledge check") changes verifier score
quality. This is the retrieval prototype to try before committing to a
fine-tuned query-generation model — see run_verifier_cot.py's module
docstring for the broader verifier-in-the-loop context.

Query construction happens ONLY from the VLM's own statement text — never
from the image's ground-truth class — otherwise "verification" becomes
circular (grounding on the answer key). Two term sources, combined:
  1. Class-vocabulary mentions: the full label set for the item's OWN
     dataset (not just the correct class) checked as substrings against the
     statement — catches an explicit diagnosis claim like "early blight"
     verbatim, and still contributes component words if only part of a
     multi-word class name appears.
  2. Generic keyword extraction: tokenize the statement and score by
     inverse document frequency against the corpus at search time, so
     generic captioning words ("leaf", "plant", "shows") don't drown out
     specific ones ("chlorosis", "Fusarium") without needing a hand-curated
     agricultural stopword list.

Retrieval is a simple inverted-index keyword lookup (word -> passage
indices), not embeddings — intentionally, to keep this a fast/cheap first
pass. If this shows a real signal, the natural next step is fine-tuning a
model to GENERATE better search queries/synonyms (a more supervisable
target than the direct-confidence-score fine-tuning already tried and found
to fail — see outputs/verifier_analysis/verifier_findings.pdf) rather than
moving to embeddings.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]{2,}")

# Small, generic English stopword list only — deliberately NOT an
# agricultural-domain list (no "leaf"/"plant"/"symptom" etc.), since those
# get downweighted anyway by the IDF term in search_passages() once they're
# indexed. A hand-curated domain stopword list would need constant upkeep as
# vocabulary shifts across datasets; IDF doesn't.
_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "this", "that",
    "from", "has", "have", "had", "not", "but", "you", "your", "its",
    "his", "her", "their", "our", "can", "could", "may", "might", "will",
    "would", "should", "there", "here", "what", "which", "who", "whom",
    "into", "onto", "than", "then", "them", "they", "these", "those",
    "been", "being", "does", "did", "doing", "over", "under", "about",
    "some", "any", "all", "each", "more", "most", "such", "only", "also",
    "very", "just", "also", "when", "where", "how", "why", "image",
    "photo", "picture",
}


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "") if w.lower() not in _STOPWORDS]


class GardianIndex:
    def __init__(self, records: list[dict], inverted: dict[str, list[int]]):
        self.records = records
        self.inverted = inverted


def load_gardian_index(path: str | Path, max_records: int | None = None) -> GardianIndex:
    """Loads the GARDIAN passages jsonl into memory and builds a word ->
    [record_index, ...] inverted index over passage + title + keywords.
    max_records caps how many lines are read (for fast prototyping on a
    subset before committing to indexing the full 690k-chunk file)."""
    records: list[dict] = []
    inverted: dict[str, list[int]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if max_records is not None and line_no >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = len(records)
            records.append(rec)
            words = set(_tokenize(rec.get("passage", "")))
            words.update(_tokenize(rec.get("title", "")))
            words.update(kw.lower() for kw in (rec.get("keywords") or []))
            for w in words:
                inverted[w].append(idx)
    print(f"  [retrieval] loaded {len(records)} passages, {len(inverted)} unique index terms "
          f"from {path}")
    return GardianIndex(records, dict(inverted))


def extract_query_terms(
    statement: str, class_vocab: list[str] | None, max_terms: int = 12,
) -> list[tuple[str, float]]:
    """Query terms derived ONLY from `statement` (the VLM's own text) — never
    from ground truth. `class_vocab` is the full label set for the item's
    dataset, used only to recognize when the statement itself names one of
    those classes (an explicit-claim signal, weighted higher than generic
    tokenized words). Returns [(term, weight), ...] sorted by weight desc."""
    statement_l = (statement or "").lower()
    terms: dict[str, float] = {}

    for cls in class_vocab or []:
        cls_l = cls.lower().strip()
        if len(cls_l) < 3:
            continue
        if cls_l in statement_l:
            terms[cls_l] = max(terms.get(cls_l, 0.0), 3.0)
        else:
            for w in _tokenize(cls_l):
                if w in statement_l:
                    terms[w] = max(terms.get(w, 0.0), 2.0)

    for w in _tokenize(statement):
        terms.setdefault(w, 1.0)

    ranked = sorted(terms.items(), key=lambda kv: -kv[1])
    return ranked[:max_terms]


def search_passages(
    index: GardianIndex, query_terms: list[tuple[str, float]], top_k: int = 3,
) -> list[dict]:
    """Scores candidate passages by sum(term_weight * idf(term)) over terms
    that actually hit the index, idf = log(1 + n_docs / (1 + doc_freq)) so
    terms that appear in most of the corpus (generic words that slipped past
    tokenization) contribute little regardless of query weight."""
    if not query_terms or not index.records:
        return []
    n_docs = max(len(index.records), 1)
    scores: dict[int, float] = defaultdict(float)
    for term, weight in query_terms:
        doc_ids = index.inverted.get(term)
        if not doc_ids:
            continue
        idf = math.log(1 + n_docs / (1 + len(doc_ids)))
        for doc_id in doc_ids:
            scores[doc_id] += weight * idf

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    return [{**index.records[doc_id], "_score": round(score, 3)} for doc_id, score in ranked]


def format_passages_for_prompt(passages: list[dict], max_chars: int = 600) -> str:
    if not passages:
        return "(no relevant reference passages found)"
    blocks = []
    for i, p in enumerate(passages, 1):
        text = (p.get("passage") or "")[:max_chars]
        title = p.get("title") or "(untitled)"
        source = p.get("source") or "unknown"
        blocks.append(f"[{i}] ({source} — \"{title}\")\n{text}")
    return "\n\n".join(blocks)
