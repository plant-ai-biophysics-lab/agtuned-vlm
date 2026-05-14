"""Build a JSON reference database from Wikipedia, PubMed, CrossRef, and CGIAR HuggingFace."""

import argparse
import csv
import json
import logging
import time
import warnings
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

EXCERPT_MAX  = 800
TIMEOUT      = 15
SLEEP        = 0.5

# PubMed journal filters — these journals are indexed in PubMed and return full abstracts
JOURNAL_FILTERS = [
    ("Plant Phenomics",                       "Plant Phenomics[Journal]"),
    ("Computers and Electronics in Agriculture", "Comput Electron Agric[Journal]"),
]

# CrossRef ISSNs for article discovery (titles/DOIs even when abstract not in PubMed)
CROSSREF_ISSNS = [
    ("Plant Phenomics",                        "2643-6515"),
    ("Computers and Electronics in Agriculture", "0168-1699"),
]

_HEADERS = {"User-Agent": "AgMLResearch/1.0 (research; academic non-commercial use)"}


def _make_key(agml_dataset: str, class_label: str, crop: str) -> str:
    def _slug(s):
        return s.strip().lower().replace(" ", "_")
    return f"{_slug(agml_dataset)}__{_slug(class_label)}__{_slug(crop)}"


def _search_query(class_label: str, crop: str, task_type: str) -> str:
    if task_type in ("weed_identification", "crop_identification"):
        return f"{class_label} plant weed visual"
    return f"{class_label} {crop} plant disease symptoms"


# ── Source 1: Wikipedia ────────────────────────────────────────────────────────

def _fetch_wikipedia(class_label: str, crop: str, task_type: str) -> list[dict]:
    query   = _search_query(class_label, crop, task_type)
    sources = []
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query,
                    "format": "json", "srlimit": 3},
            headers=_HEADERS, timeout=TIMEOUT,
        )
        resp.raise_for_status()
        time.sleep(SLEEP)
        for hit in resp.json().get("query", {}).get("search", [])[:2]:
            title = hit["title"]
            try:
                ex = requests.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={"action": "query", "titles": title, "prop": "extracts",
                            "exintro": True, "format": "json", "explaintext": True},
                    headers=_HEADERS, timeout=TIMEOUT,
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


# ── Source 2: PubMed general + journal-filtered ────────────────────────────────

def _pubmed_fetch(term: str, organization: str, extra: dict | None = None) -> list[dict]:
    """Run one PubMed esearch + efetch, return source dicts."""
    sources = []
    try:
        ids = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": term, "retmax": 2, "retmode": "json"},
            headers=_HEADERS, timeout=TIMEOUT,
        )
        ids.raise_for_status()
        time.sleep(SLEEP)
        pmids = ids.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return sources

        fetch = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"},
            headers=_HEADERS, timeout=TIMEOUT,
        )
        fetch.raise_for_status()
        time.sleep(SLEEP)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            soup = BeautifulSoup(fetch.text, "lxml-xml")

        for art in soup.find_all("PubmedArticle"):
            pmid     = (art.find("PMID") or soup.new_tag("x")).get_text(strip=True)
            title    = (art.find("ArticleTitle") or soup.new_tag("x")).get_text(strip=True)
            abstract = (art.find("AbstractText") or soup.new_tag("x")).get_text(strip=True)
            journal  = (art.find("Title") or soup.new_tag("x")).get_text(strip=True)
            if not abstract:
                continue
            entry = {
                "organization":  organization,
                "title":         title,
                "journal":       journal,
                "pmid":          pmid,
                "url":           f"https://pubmed.ncbi.nlm.nih.gov/{pmid}",
                "type":          "journal_abstract",
                "accessed_date": str(date.today()),
                "excerpt":       abstract[:EXCERPT_MAX],
                "search_query":  term,
            }
            if extra:
                entry.update(extra)
            sources.append(entry)
    except Exception as e:
        log.warning("PubMed '%s': %s", term, e)
    return sources


def _fetch_pubmed(class_label: str, crop: str, task_type: str) -> list[dict]:
    query   = _search_query(class_label, crop, task_type)
    sources = _pubmed_fetch(f"{query}[Title/Abstract]", "PubMed")
    # Journal-specific queries for Plant Phenomics and C&E in Ag
    for journal_display, journal_filter in JOURNAL_FILTERS:
        sources += _pubmed_fetch(
            f"{query}[Title/Abstract] AND {journal_filter}",
            organization=journal_display,
        )
    return sources


# ── Source 3: CrossRef (article discovery / provenance) ───────────────────────

def _fetch_crossref(class_label: str, crop: str, task_type: str) -> list[dict]:
    """Searches CrossRef for titles and DOIs. Abstracts often absent; records kept for provenance."""
    query   = _search_query(class_label, crop, task_type)
    sources = []
    for journal_name, issn in CROSSREF_ISSNS:
        try:
            resp = requests.get(
                "https://api.crossref.org/works",
                params={
                    "query":  query,
                    "filter": f"issn:{issn}",
                    "rows":   2,
                    "select": "DOI,title,abstract,published,container-title",
                },
                headers=_HEADERS, timeout=TIMEOUT,
            )
            resp.raise_for_status()
            time.sleep(SLEEP)
            for item in resp.json().get("message", {}).get("items", []):
                title    = (item.get("title") or [""])[0]
                doi      = item.get("DOI", "")
                # Strip JATS XML tags if abstract present
                raw_abs  = item.get("abstract", "")
                abstract = BeautifulSoup(raw_abs, "html.parser").get_text().strip() if raw_abs else ""
                sources.append({
                    "organization":  journal_name,
                    "title":         title,
                    "doi":           doi,
                    "url":           f"https://doi.org/{doi}" if doi else "",
                    "type":          "journal_article",
                    "journal":       (item.get("container-title") or [""])[0],
                    "accessed_date": str(date.today()),
                    "excerpt":       abstract[:EXCERPT_MAX] if abstract else "",
                    "search_query":  query,
                })
        except Exception as e:
            log.warning("CrossRef '%s' (%s): %s", query, journal_name, e)
    return sources


# ── Source 4: CGIAR HuggingFace dataset cards ─────────────────────────────────

def _load_cgiar_cache() -> dict[str, str]:
    """Fetch all CGIAR HF dataset READMEs once. Returns {dataset_id: readme_text}."""
    cache: dict[str, str] = {}
    try:
        resp = requests.get(
            "https://huggingface.co/api/datasets",
            params={"author": "CGIAR", "limit": 100},
            headers=_HEADERS, timeout=TIMEOUT,
        )
        resp.raise_for_status()
        datasets = resp.json()
    except Exception as e:
        log.warning("CGIAR HF dataset list: %s", e)
        return cache

    for ds in datasets:
        ds_id = ds.get("id", "")
        if not ds_id:
            continue
        try:
            r = requests.get(
                f"https://huggingface.co/datasets/{ds_id}/raw/main/README.md",
                headers=_HEADERS, timeout=TIMEOUT,
            )
            if r.status_code == 200:
                cache[ds_id] = r.text
            time.sleep(SLEEP)
        except Exception as e:
            log.warning("CGIAR HF README '%s': %s", ds_id, e)
    log.info("CGIAR HF cache: %d dataset READMEs loaded", len(cache))
    return cache


def _search_cgiar(class_label: str, crop: str, cgiar_cache: dict[str, str]) -> list[dict]:
    """Search cached CGIAR READMEs for class/crop mentions."""
    terms   = [t.lower() for t in [class_label, crop] if len(t) > 3]
    sources = []
    for ds_id, readme in cgiar_cache.items():
        readme_lower = readme.lower()
        if not any(t in readme_lower for t in terms):
            continue
        for term in terms:
            idx = readme_lower.find(term)
            if idx < 0:
                continue
            start   = max(0, idx - 150)
            end     = min(len(readme), idx + 450)
            excerpt = readme[start:end].strip()
            sources.append({
                "organization":  "CGIAR",
                "title":         ds_id,
                "url":           f"https://huggingface.co/datasets/{ds_id}",
                "type":          "dataset_card",
                "accessed_date": str(date.today()),
                "excerpt":       excerpt[:EXCERPT_MAX],
                "search_query":  f"{class_label} {crop}",
            })
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

    print(f"Loading CGIAR HuggingFace dataset cache...")
    cgiar_cache = _load_cgiar_cache()

    entries = dict(existing_entries)
    total   = len(seen)

    for n, (key, meta) in enumerate(seen.items(), 1):
        if args.skip_existing and key in existing_entries:
            print(f"  {n}/{total} [skip] {meta['class_label']} ({meta['crop']})")
            continue

        sources  = (
            _fetch_wikipedia(meta["class_label"], meta["crop"], meta["task_type"])
            + _fetch_pubmed(meta["class_label"], meta["crop"], meta["task_type"])
            + _fetch_crossref(meta["class_label"], meta["crop"], meta["task_type"])
            + _search_cgiar(meta["class_label"], meta["crop"], cgiar_cache)
        )
        status = "found" if any(s.get("excerpt") for s in sources) else "no_source_found"

        entries[key] = {**meta, "status": status, "sources": sources}
        print(f"  {n}/{total} {meta['class_label']} ({meta['crop']}) — {status} "
              f"[{len(sources)} sources]")

        dst.write_text(
            json.dumps({"schema_version": "1.1", "built_date": str(date.today()),
                        "entries": entries}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    found = sum(1 for e in entries.values() if e.get("status") == "found")
    print(f"\nDone. {found}/{len(entries)} entries have at least one source → {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build reference DB from Wikipedia, PubMed, CrossRef, and CGIAR HuggingFace."
    )
    parser.add_argument("--descriptions_csv", default="outputs/descriptions_export.csv")
    parser.add_argument("--output_db",        default="outputs/reference_db.json")
    parser.add_argument("--skip_existing",    action="store_true",
                        help="Skip entries already present in an existing reference_db.json")
    main(parser.parse_args())
