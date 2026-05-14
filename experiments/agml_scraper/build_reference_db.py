"""Build a JSON reference database from Wikipedia, USDA APHIS IDtools, and CGIAR HuggingFace."""

import argparse
import csv
import json
import logging
import re
import time
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

EXCERPT_MAX  = 800
TIMEOUT      = 15
SLEEP        = 0.5
WIKI_RETRY_DELAYS = [2, 5, 10]  # backoff on 429

# Class labels that indicate a healthy/normal sample — skip disease-symptom queries
_HEALTHY_LABELS = {"healthy", "normal", "no disease", "no stress", "unaffected"}

_HEADERS = {"User-Agent": "AgMLResearch/1.0 (research; academic non-commercial use)"}
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AgMLResearch/1.0; "
        "academic data collection for agricultural AI research)"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


def _make_key(agml_dataset: str, class_label: str, crop: str) -> str:
    def _slug(s):
        return s.strip().lower().replace(" ", "_")
    return f"{_slug(agml_dataset)}__{_slug(class_label)}__{_slug(crop)}"


def _is_healthy(class_label: str) -> bool:
    return class_label.strip().lower() in _HEALTHY_LABELS


def _search_query(class_label: str, crop: str, task_type: str) -> str:
    if task_type in ("weed_identification", "crop_identification"):
        return f"{class_label} plant weed visual"
    return f"{class_label} {crop} plant disease symptoms"


def _wiki_get(url: str, params: dict) -> requests.Response:
    """GET with automatic retry on 429."""
    for attempt, delay in enumerate([0] + WIKI_RETRY_DELAYS):
        if delay:
            log.info("Wikipedia 429 — retrying in %ds (attempt %d)", delay, attempt + 1)
            time.sleep(delay)
        r = requests.get(url, params=params, headers=_HEADERS, timeout=TIMEOUT)
        if r.status_code != 429:
            return r
    return r  # return final response even if still 429


# ── Source 1: Wikipedia ────────────────────────────────────────────────────────

def _fetch_wikipedia(class_label: str, crop: str, task_type: str) -> list[dict]:
    if _is_healthy(class_label):
        return []
    query   = _search_query(class_label, crop, task_type)
    sources = []
    try:
        resp = _wiki_get(
            "https://en.wikipedia.org/w/api.php",
            {"action": "query", "list": "search", "srsearch": query,
             "format": "json", "srlimit": 3},
        )
        resp.raise_for_status()
        time.sleep(SLEEP)
        for hit in resp.json().get("query", {}).get("search", [])[:2]:
            title = hit["title"]
            try:
                ex = _wiki_get(
                    "https://en.wikipedia.org/w/api.php",
                    {"action": "query", "titles": title, "prop": "extracts",
                     "exintro": True, "format": "json", "explaintext": True},
                )
                ex.raise_for_status()
                time.sleep(SLEEP)
                pages   = ex.json().get("query", {}).get("pages", {})
                extract = next(iter(pages.values())).get("extract", "").strip()
                if not extract:
                    continue
                sources.append({
                    "organization":  "Wikipedia",
                    "title":         title,
                    "url":           f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    "type":          "encyclopedia",
                    "accessed_date": str(date.today()),
                    "excerpt":       extract[:EXCERPT_MAX],
                    "search_query":  query,
                })
            except Exception as e:
                log.warning("Wikipedia extract '%s': %s", title, e)
    except Exception as e:
        log.warning("Wikipedia search '%s': %s", query, e)
    return sources


# ── Source 2: USDA APHIS IDtools ───────────────────────────────────────────────

# Package index pages that list all individual fact sheet links
_IDTOOLS_PACKAGE_INDEXES = [
    "https://idtools.org/citrus_pests/index.cfm?packageID=63",
    "https://idtools.org/citrus_diseases/index.cfm?packageID=14",
    "https://idtools.org/phytophthora/index.cfm?packageID=1131",
    "https://idtools.org/aphid/index.cfm",
    "https://idtools.org/thrips/index.cfm",
    "https://idtools.org/sawfly/index.cfm",
    "https://idtools.org/tortricid/index.cfm",
]

# Section headings that contain visual / identification content
_VISUAL_HEADINGS = {
    "symptom", "symptoms", "description", "identification", "visual",
    "appearance", "morphology", "diagnostic", "characteristics",
    "field identification", "signs", "damage symptoms", "overview",
}

# Section headings to skip
_SKIP_HEADINGS = {
    "management", "control", "chemical control", "biological control",
    "treatment", "biology", "life cycle", "distribution", "references",
    "links", "contact", "navigation", "footer", "disclaimer",
}


def _load_idtools_index(delay: float = 1.0) -> dict[str, dict]:
    """Crawl IDtools package index pages → {subject_lower: {url, subject}} map."""
    index: dict[str, dict] = {}

    for seed_url in _IDTOOLS_PACKAGE_INDEXES:
        try:
            r = requests.get(seed_url, headers=_BROWSER_HEADERS, timeout=TIMEOUT)
            if r.status_code != 200:
                log.debug("IDtools index %s → HTTP %d", seed_url, r.status_code)
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            base = r.url
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href or href.startswith(("#", "mailto:", "javascript:")):
                    continue
                abs_url = urljoin(base, href).split("#")[0]
                if "idtools.org" not in abs_url or "entityID" not in abs_url:
                    continue
                subject = a.get_text(strip=True)
                if subject and 3 < len(subject) < 100:
                    index[subject.lower()] = {"url": abs_url, "subject": subject}
            time.sleep(delay)
        except Exception as e:
            log.warning("IDtools index '%s': %s", seed_url, e)

    log.info("IDtools index: %d fact sheets indexed", len(index))
    return index


def _scrape_idtools_factsheet(url: str) -> str:
    """Fetch an IDtools fact sheet and extract visual description text."""
    try:
        r = requests.get(url, headers=_BROWSER_HEADERS, timeout=TIMEOUT)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        visual_parts = []
        for heading in soup.find_all(re.compile(r"^h[1-4]$")):
            h_text = heading.get_text(strip=True).lower().rstrip(":")
            h_clean = re.sub(r"[^a-z ]", "", h_text).strip()

            if any(skip in h_clean for skip in _SKIP_HEADINGS):
                continue
            if not any(vis in h_clean for vis in _VISUAL_HEADINGS):
                continue

            section_parts = []
            for sib in heading.find_next_siblings():
                if sib.name and re.match(r"^h[1-4]$", sib.name):
                    break
                text = sib.get_text(separator=" ", strip=True)
                if text and len(text) > 15:
                    section_parts.append(text)

            if section_parts:
                visual_parts.append(" ".join(section_parts))

        return " ".join(visual_parts).strip()
    except Exception as e:
        log.warning("IDtools scrape '%s': %s", url, e)
        return ""


def _search_idtools(
    class_label: str, crop: str, idtools_index: dict[str, dict]
) -> list[dict]:
    """Match class label against IDtools subject index; scrape visual text from match."""
    if not idtools_index:
        return []

    terms = [t.lower() for t in class_label.split() if len(t) > 3]
    if not terms:
        return []

    matches = [
        entry for subj, entry in idtools_index.items()
        if any(t in subj for t in terms)
    ]
    if not matches:
        return []

    sources = []
    for match in matches[:1]:
        excerpt = _scrape_idtools_factsheet(match["url"])
        if not excerpt:
            continue
        sources.append({
            "organization":  "USDA APHIS IDtools",
            "title":         match["subject"],
            "url":           match["url"],
            "type":          "fact_sheet",
            "accessed_date": str(date.today()),
            "excerpt":       excerpt[:EXCERPT_MAX],
            "search_query":  f"{class_label} {crop}",
        })
        time.sleep(SLEEP)

    return sources


# ── Source 3: CGIAR GARDIAN research document chunks ──────────────────────────
# Uses the two public GARDIAN QA datasets which contain ~44K context chunks
# extracted from CGIAR research publications (288-word windows, full text).

_CGIAR_DATASETS = [
    "CGIAR/ragas_gardian_evaluation_non_overlapping",
    "CGIAR/ragas_gardian_evaluation_overlapping",
]


def _load_cgiar_cache() -> list[str]:
    """Load CGIAR GARDIAN context chunks for in-memory keyword search."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("'datasets' package not installed — CGIAR source disabled")
        return []

    contexts: list[str] = []
    for ds_id in _CGIAR_DATASETS:
        try:
            ds = load_dataset(ds_id, split="train", trust_remote_code=True)
            before = len(contexts)
            for row in ds:
                ctx = row.get("context") or ""
                if ctx:
                    contexts.append(ctx)
            log.info("CGIAR %s: %d chunks loaded", ds_id, len(contexts) - before)
        except Exception as e:
            log.warning("CGIAR load '%s': %s", ds_id, e)

    log.info("CGIAR total: %d document chunks cached", len(contexts))
    return contexts


def _search_cgiar(class_label: str, crop: str, cgiar_contexts: list[str]) -> list[dict]:
    """Search CGIAR GARDIAN document chunks for class/crop mentions."""
    if not cgiar_contexts:
        return []
    terms = [t.lower() for t in [class_label, crop] if len(t) > 3]
    if not terms:
        return []

    sources = []
    for ctx in cgiar_contexts:
        if not any(t in ctx.lower() for t in terms):
            continue
        sources.append({
            "organization":  "CGIAR GARDIAN",
            "title":         "CGIAR GARDIAN Research Corpus",
            "url":           "https://huggingface.co/datasets/CGIAR/ragas_gardian_evaluation_non_overlapping",
            "type":          "research_document",
            "accessed_date": str(date.today()),
            "excerpt":       ctx[:EXCERPT_MAX],
            "search_query":  f"{class_label} {crop}",
        })
        if len(sources) >= 2:
            break
    return sources


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    src = Path(args.descriptions_csv)
    dst = Path(args.output_db)
    dst.parent.mkdir(parents=True, exist_ok=True)

    existing_db      = json.loads(dst.read_text(encoding="utf-8")) if dst.exists() else {}
    existing_entries = existing_db.get("entries", {})

    seen: dict[str, dict] = {}
    with open(src, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = _make_key(row["agml_dataset"], row["class_label"], row["crop"])
            if key not in seen:
                seen[key] = {k: row[k] for k in ("agml_dataset", "crop", "class_label", "task_type")}

    print("Loading USDA APHIS IDtools index...")
    idtools_index = _load_idtools_index()

    print("Loading CGIAR GARDIAN document chunks (may take a moment on first run)...")
    cgiar_contexts = _load_cgiar_cache()

    entries = dict(existing_entries)
    total   = len(seen)

    for n, (key, meta) in enumerate(seen.items(), 1):
        if args.skip_existing and key in existing_entries:
            print(f"  {n}/{total} [skip] {meta['class_label']} ({meta['crop']})")
            continue

        if _is_healthy(meta["class_label"]):
            entries[key] = {**meta, "status": "healthy_skip", "sources": []}
            print(f"  {n}/{total} [healthy] {meta['class_label']} ({meta['crop']}) — skipped")
            dst.write_text(
                json.dumps({"schema_version": "1.2", "built_date": str(date.today()),
                            "entries": entries}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            continue

        sources = (
            _fetch_wikipedia(meta["class_label"], meta["crop"], meta["task_type"])
            + _search_idtools(meta["class_label"], meta["crop"], idtools_index)
            + _search_cgiar(meta["class_label"], meta["crop"], cgiar_contexts)
        )
        status = "found" if any(s.get("excerpt") for s in sources) else "no_source_found"

        entries[key] = {**meta, "status": status, "sources": sources}
        print(f"  {n}/{total} {meta['class_label']} ({meta['crop']}) — {status} "
              f"[{len(sources)} sources]")

        dst.write_text(
            json.dumps({"schema_version": "1.2", "built_date": str(date.today()),
                        "entries": entries}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    found = sum(1 for e in entries.values() if e.get("status") == "found")
    print(f"\nDone. {found}/{len(entries)} entries have at least one source → {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build reference DB from Wikipedia, USDA APHIS IDtools, and CGIAR HuggingFace."
    )
    parser.add_argument("--descriptions_csv", default="outputs/descriptions_export.csv")
    parser.add_argument("--output_db",        default="outputs/reference_db.json")
    parser.add_argument("--skip_existing",    action="store_true",
                        help="Skip entries already present in an existing reference_db.json")
    main(parser.parse_args())
