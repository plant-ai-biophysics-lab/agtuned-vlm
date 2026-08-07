"""
Stage 1a: Build a JSON reference database of grounding sources per class.

For every class across the active datasets in datasets.yaml, fetches:
  1. Wikipedia (always) — full article text, not just the intro, for the
     top 3 matching pages.
  2. Live web search (DuckDuckGo via `ddgs`, no API key) — only as a
     FALLBACK when Wikipedia comes back empty or too thin. `ddgs` fans a
     single query out across several search engines' HTML (Yandex, Google,
     Brave, Mojeek, Grokipedia, none of them official APIs), so it's the
     part of this script most likely to get rate-limited/blocked at scale;
     restricting it to a fallback keeps that risk to only the classes that
     actually need it, since Wikipedia alone resolves most of them.

No pre-chunked corpus (CGIAR GARDIAN) and no local scraper DB (USDA APHIS
IDtools) — both retired in favor of live per-class search: neither
guaranteed a source that actually discussed the *specific* class, and a
fresh, per-class web search plus a relevance gate (the page must mention the
class label, not just rank well for the query) gets closer to that than a
static pre-built corpus can.

Runs BEFORE generate.py now, not after: this reads directly from
datasets.yaml rather than from a descriptions JSONL generate.py already
produced, and generate.py's synthesis pass now reads THIS file as its only
input material (no vision-captioning pass — descriptions are synthesized
purely from what's fetched here).

Results are written incrementally to reference_db.json so runs are safely resumable.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import logging
import re
import time
from datetime import date

import requests
import trafilatura
from ddgs import DDGS

from src.common.dataset_config import active_datasets, parse_class_label
from src.common.io import save_json
from src.common.progress import ProgressBar

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

EXCERPT_MAX = 800
TIMEOUT = 15
WIKI_SLEEP = 2.0
WIKI_RETRY_DELAYS = [5, 15, 30]
WIKI_MAX_PAGES = 3           # top matching Wikipedia pages to pull full text from
WEB_SEARCH_MAX_RESULTS = 5   # candidate URLs pulled per class
WEB_SEARCH_KEEP = 3          # how many relevant ones to keep as sources
WEB_FETCH_SLEEP = 1.0
WIKI_THIN_CHARS = 400        # combined Wikipedia excerpt length below this
                              # triggers the web-search fallback

_HEALTHY_LABELS = {"healthy", "normal", "no disease", "no stress", "unaffected",
                    "disease free", "disease-free", "no virus", "no defect"}
_HEALTHY_WORDS = {"healthy", "heathy", "healty", "normal", "unaffected"}  # incl.
                  # common dataset-label typos actually observed (heathy, healty)
_QUALIFIER_WORDS = {  # crop/plant-part words that appear alongside a healthy
    # label but don't change its meaning — "Healthy Rice Leaf" is exactly as
    # groundless as "Healthy"
    "leaf", "leaves", "plant", "fruit", "sample", "rice", "brinjal",
}
_HEADERS = {"User-Agent": "AgMLResearch/1.0 (research; academic non-commercial use)"}


def _make_key(agml_dataset: str, class_label: str, crop: str) -> str:
    def _slug(s):
        return s.strip().lower().replace(" ", "_")
    return f"{_slug(agml_dataset)}__{_slug(class_label)}__{_slug(crop)}"


_PAREN_RE = re.compile(r"^(.*?)\s*\(([^)]+)\)\s*$")


def _split_parens(term: str) -> list[str]:
    """[term] normally, or [term, common, scientific] if term looks like
    "Common (Scientific)" — see _match_terms."""
    m = _PAREN_RE.match(term)
    if not m:
        return [term]
    common, scientific = m.group(1).strip(), m.group(2).strip()
    parts = [term]
    if common:
        parts.append(common)
    if scientific:
        parts.append(scientific)
    return parts


def _match_terms(class_label: str, aliases: list[str] | None = None) -> list[str]:
    """Terms to try matching against fetched text, in priority order.

    Most labels are used as-is. But a label like "Datura (Datura stramonium)"
    or "Neem (Azadirachta indica)" — common name plus scientific name in
    parens — almost never appears verbatim in real text; a page about neem is
    overwhelmingly likely to say "Neem" or "Azadirachta indica" on its own,
    not the exact combined string. Splitting into candidates and accepting a
    match on either recovers those without weakening the gate for ordinary
    single-part labels (still an exact-ish substring match, just tried
    against more candidate strings).

    aliases (from datasets.yaml's class_aliases, see effective_search_terms)
    are folded in too, parenthetical-split the same way — a source found via
    an alias query should still count as a match if it mentions the raw
    class_label, and vice versa."""
    terms = list(_split_parens(class_label))
    for alias in (aliases or []):
        for t in _split_parens(alias):
            if t not in terms:
                terms.append(t)
    return terms


def _find_match(text: str, terms: list[str]) -> str | None:
    """First term (in priority order) that appears in text, or None."""
    text_lower = text.lower()
    for term in terms:
        if term.lower() in text_lower:
            return term
    return None


def _excerpt_around_match(text: str, term: str, max_len: int) -> str:
    """First max_len chars starting a bit before term's first occurrence, so
    the relevance match that got this source accepted is actually inside the
    saved excerpt — the source's full text can be much longer than max_len,
    and a match late in it would otherwise fall outside a naive text[:max_len]
    slice."""
    idx = text.lower().find(term.lower())
    if idx == -1:
        return text[:max_len]
    start = max(0, idx - max_len // 4)
    return text[start:start + max_len]


def _is_healthy(class_label: str) -> bool:
    label = class_label.strip().lower()
    if label in _HEALTHY_LABELS:
        return True
    # Tokenized fallback: "Healthy Rice Leaf" / "Healty Brinjal" / "Heathy
    # Leaf" all mean exactly what "Healthy" alone does — once the
    # crop/plant-part qualifier words are stripped, what's left should be
    # nothing but a healthy word (possibly repeated/typo'd), not a mix of a
    # healthy word and something else that would actually need grounding.
    tokens = [t for t in re.split(r"[\s\-]+", label) if t]
    core = [t for t in tokens if t not in _QUALIFIER_WORDS]
    return bool(core) and all(t in _HEALTHY_WORDS for t in core)


def _search_query(class_label: str, crop: str, task_type: str) -> str:
    if task_type in ("species_identification", "quality_identification"):
        return f"{class_label} {crop} plant visual identification"
    return f"{class_label} {crop} plant disease symptoms"


def _wiki_get(url: str, params: dict) -> requests.Response:
    """GET with polite sleep and automatic retry on 429."""
    for attempt, delay in enumerate([0] + WIKI_RETRY_DELAYS):
        if delay:
            log.debug("Wikipedia 429 — retrying in %ds (attempt %d)", delay, attempt + 1)
            time.sleep(delay)
        r = requests.get(url, params=params, headers=_HEADERS, timeout=TIMEOUT)
        time.sleep(WIKI_SLEEP)
        if r.status_code != 429:
            return r
    log.debug("Wikipedia still 429 after %d retries — skipping", len(WIKI_RETRY_DELAYS))
    return r


def _fetch_wikipedia(search_term: str, crop: str, task_type: str, match_terms: list[str]) -> list[dict]:
    """search_term is what's actually queried; match_terms (from _match_terms,
    already includes search_term itself) is what counts as a relevant hit —
    kept separate so an alias query can still credit a match against the
    original raw class_label, or another alias, not just itself."""
    query = _search_query(search_term, crop, task_type)
    terms = match_terms
    sources = []
    try:
        resp = _wiki_get(
            "https://en.wikipedia.org/w/api.php",
            {"action": "query", "list": "search", "srsearch": query,
             "format": "json", "srlimit": 3},
        )
        if resp.status_code != 200:
            return sources
        for hit in resp.json().get("query", {}).get("search", [])[:WIKI_MAX_PAGES]:
            title = hit["title"]
            try:
                ex = _wiki_get(
                    "https://en.wikipedia.org/w/api.php",
                    {"action": "query", "titles": title, "prop": "extracts",
                     "format": "json", "explaintext": True},
                )
                if ex.status_code != 200:
                    continue
                pages = ex.json().get("query", {}).get("pages", {})
                extract = next(iter(pages.values())).get("extract", "").strip()
                if not extract:
                    continue
                matched = _find_match(extract, terms)
                if matched is None:
                    continue  # relevance gate — same rule as _fetch_web_search:
                              # a page ranking well for the query doesn't mean
                              # it's actually about this specific class
                sources.append({
                    "organization": "Wikipedia",
                    "title": title,
                    "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    "type": "encyclopedia",
                    "accessed_date": str(date.today()),
                    "excerpt": _excerpt_around_match(extract, matched, EXCERPT_MAX),
                    "search_query": query,
                })
            except Exception as e:
                log.debug("Wikipedia extract '%s': %s", title, e)
    except Exception as e:
        log.debug("Wikipedia search '%s': %s", query, e)
    return sources


def _fetch_web_search(search_term: str, crop: str, task_type: str, match_terms: list[str]) -> list[dict]:
    """Live web search, no pre-built corpus: search DuckDuckGo for search_term,
    fetch each candidate page fresh, extract clean article text with
    trafilatura, and keep only pages that match one of match_terms — a
    relevance gate on top of the search engine's own ranking, since a page
    ranking well for the query doesn't guarantee it's actually about this
    specific class."""
    query = _search_query(search_term, crop, task_type)
    terms = match_terms
    sources: list[dict] = []
    try:
        hits = DDGS().text(query, max_results=WEB_SEARCH_MAX_RESULTS)
    except Exception as e:
        log.debug("Web search '%s': %s", query, e)
        return sources

    for hit in hits:
        if len(sources) >= WEB_SEARCH_KEEP:
            break
        url = hit.get("href")
        if not url:
            continue
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                continue
            text = trafilatura.extract(downloaded)
            if not text:
                continue
            matched = _find_match(text, terms)
            if matched is None:
                continue  # relevance gate — page must actually mention the class
            sources.append({
                "organization": "Web search",
                "title": hit.get("title") or url,
                "url": url,
                "type": "web_page",
                "accessed_date": str(date.today()),
                "excerpt": _excerpt_around_match(text, matched, EXCERPT_MAX),
                "search_query": query,
            })
        except Exception as e:
            log.debug("Fetch '%s': %s", url, e)
        time.sleep(WEB_FETCH_SLEEP)
    return sources


def fetch_sources_for_class(
    class_label: str, crop: str, task_type: str, search_terms: list[str],
) -> tuple[list[dict], str]:
    """Try each term in search_terms (class_label first, then any
    datasets.yaml class_aliases, in order) as the actual search query —
    Wikipedia, then web-search fallback if thin — stopping at the first term
    that finds anything. match_terms (what counts as a relevant hit) is the
    same regardless of which term is being queried: the raw class_label plus
    every alias, parenthetical-split — a source found via an alias query
    should still count as a match if it mentions the original raw label.

    Returns (sources, term_used) — term_used is "" if nothing was found with
    any term, otherwise whichever one succeeded (useful for the [alias: ...]
    note in the per-class print line)."""
    match_terms: list[str] = []
    for t in search_terms:
        for part in _split_parens(t):
            if part not in match_terms:
                match_terms.append(part)

    for term in search_terms:
        sources = _fetch_wikipedia(term, crop, task_type, match_terms)
        wiki_chars = sum(len(s["excerpt"]) for s in sources)
        if wiki_chars < WIKI_THIN_CHARS:
            sources = sources + _fetch_web_search(term, crop, task_type, match_terms)
        if sources:
            return sources, term
    return [], ""


def main(args: argparse.Namespace) -> None:
    dst = Path(args.output_db)
    dst.parent.mkdir(parents=True, exist_ok=True)

    datasets_cfg = active_datasets(Path(args.config))
    if args.only:
        pattern = re.compile(args.only)
        datasets_cfg = [d for d in datasets_cfg if pattern.search(d.agml_name)]
        if not datasets_cfg:
            print(f"ERROR: --only {args.only!r} matched no active datasets.")
            sys.exit(1)

    seen: dict[str, dict] = {}
    for ds in datasets_cfg:
        for raw_label in ds.classes:
            crop, clean_cls = parse_class_label(raw_label, ds.crop)
            task_type = ds.effective_task_type(raw_label)
            key = _make_key(ds.agml_name, clean_cls, crop)
            if key not in seen:
                seen[key] = {
                    "agml_dataset": ds.agml_name,
                    "crop": crop,
                    "class_label": clean_cls,
                    "raw_label": raw_label,  # manifest's per-class key, needed to sample an
                                              # image back for this class (see generate.py)
                    "task_type": task_type,
                    # clean_cls first, then any datasets.yaml class_aliases
                    # (keyed by raw_label, same convention as class_overrides;
                    # effective_search_terms puts raw_label itself first,
                    # which we don't want as the primary term — drop it,
                    # keep just the aliases behind it)
                    "search_terms": [clean_cls] + ds.effective_search_terms(raw_label)[1:],
                }

    existing_db = json.loads(dst.read_text(encoding="utf-8")) if dst.exists() else {}
    existing_entries = existing_db.get("entries", {})
    entries = dict(existing_entries)

    for key, meta in ProgressBar(list(seen.items()), desc="Building reference DB"):
        prior_status = existing_entries.get(key, {}).get("status")
        if args.skip_existing and key in existing_entries and prior_status != "no_source_found":
            print(f"  [skip] {meta['class_label']} ({meta['crop']})")
            continue

        if _is_healthy(meta["class_label"]):
            entries[key] = {**meta, "status": "healthy_skip", "sources": []}
            print(f"  [healthy] {meta['class_label']} ({meta['crop']}) — skipped")
            save_json(dst, {"schema_version": "2.0", "built_date": str(date.today()), "entries": entries})
            continue

        search_terms = meta.get("search_terms") or [meta["class_label"]]
        sources, term_used = fetch_sources_for_class(
            meta["class_label"], meta["crop"], meta["task_type"], search_terms
        )
        status = "found" if sources else "no_source_found"

        entries[key] = {**meta, "status": status, "sources": sources}
        used_fallback = any(s.get("organization") == "Web search" for s in sources)
        fallback_note = " [web-search fallback]" if used_fallback else ""
        alias_note = f" [via alias: {term_used}]" if term_used and term_used != meta["class_label"] else ""
        print(f"  {meta['class_label']} ({meta['crop']}) — {status} "
              f"[{len(sources)} sources]{fallback_note}{alias_note}")

        save_json(dst, {"schema_version": "2.0", "built_date": str(date.today()), "entries": entries})

    found = sum(1 for e in entries.values() if e.get("status") == "found")
    print(f"\nDone. {found}/{len(entries)} entries have at least one source → {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a class-level reference DB from Wikipedia and live web search."
    )
    parser.add_argument(
        "--config", default="src/config/datasets.yaml",
        help="Path to datasets.yaml registry",
    )
    parser.add_argument(
        "--output_db", default="outputs/data_generation/reference_db.json",
    )
    parser.add_argument(
        "--only", default=None,
        help="optional regex filter on agml_name — build for a single dataset "
             "(or a subset) instead of the full active list. Same convention as "
             "download_and_sample.py's --only.",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip entries already present in an existing reference_db.json — "
             "except ones with status no_source_found, which are always "
             "retried (a rerun is how you recover from a matching-logic fix "
             "or a search that failed transiently, not just fill in gaps).",
    )
    main(parser.parse_args())
