"""
Stage 1b: Build a JSON reference database from Wikipedia, USDA APHIS IDtools, and CGIAR HuggingFace.

Reads descriptions_export.csv produced by generate.py and fetches reference material
for each class label from three sources:
  1. Wikipedia — introductory encyclopedia excerpts
  2. USDA APHIS IDtools — pest and disease fact sheets
  3. CGIAR GARDIAN — research publication chunks via semantic search

Results are written incrementally to reference_db.json so runs are safely resumable.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import csv
import json
import logging
import time
from datetime import date

import requests
from bs4 import BeautifulSoup

from src.common.io import save_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

EXCERPT_MAX = 800
TIMEOUT = 15
SLEEP = 0.5
WIKI_SLEEP = 2.0
WIKI_RETRY_DELAYS = [5, 15, 30]

_HEALTHY_LABELS = {"healthy", "normal", "no disease", "no stress", "unaffected"}
_HEADERS = {"User-Agent": "AgMLResearch/1.0 (research; academic non-commercial use)"}

_WEB_SCRAPER_DIR = Path(__file__).parent.parent.parent / "experiments" / "web_scraper"
_IDTOOLS_DB = _WEB_SCRAPER_DIR / "idtools_scraper.db"


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


def _fetch_wikipedia(class_label: str, crop: str, task_type: str) -> list[dict]:
    if _is_healthy(class_label):
        return []
    query = _search_query(class_label, crop, task_type)
    sources = []
    try:
        resp = _wiki_get(
            "https://en.wikipedia.org/w/api.php",
            {"action": "query", "list": "search", "srsearch": query,
             "format": "json", "srlimit": 3},
        )
        if resp.status_code != 200:
            return sources
        for hit in resp.json().get("query", {}).get("search", [])[:2]:
            title = hit["title"]
            try:
                ex = _wiki_get(
                    "https://en.wikipedia.org/w/api.php",
                    {"action": "query", "titles": title, "prop": "extracts",
                     "exintro": True, "format": "json", "explaintext": True},
                )
                if ex.status_code != 200:
                    continue
                pages = ex.json().get("query", {}).get("pages", {})
                extract = next(iter(pages.values())).get("extract", "").strip()
                if not extract:
                    continue
                sources.append({
                    "organization": "Wikipedia",
                    "title": title,
                    "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    "type": "encyclopedia",
                    "accessed_date": str(date.today()),
                    "excerpt": extract[:EXCERPT_MAX],
                    "search_query": query,
                })
            except Exception as e:
                log.debug("Wikipedia extract '%s': %s", title, e)
    except Exception as e:
        log.debug("Wikipedia search '%s': %s", query, e)
    return sources


def _load_idtools_index() -> dict[str, dict]:
    """
    Build {subject_lower: {url, subject}} map using scrape_idtools.py infrastructure.

    If idtools_scraper.db already exists: read subjects from SQLite instantly.
    If not: run URL discovery (same crawler as scrape_idtools.py) and save to DB.
    """
    import sys
    sys.path.insert(0, str(_WEB_SCRAPER_DIR))
    try:
        from scrape_idtools import init_db
    except ImportError as e:
        log.warning("Cannot import scrape_idtools — IDtools disabled: %s", e)
        return {}

    conn = init_db(str(_IDTOOLS_DB))
    known = conn.execute("SELECT COUNT(*) FROM urls").fetchone()[0]

    if known < 50:
        log.warning(
            "IDtools DB is empty. Run discovery first:\n"
            "  cd experiments/web_scraper && uv run python scrape_idtools.py --discover-only\n"
            "Skipping IDtools source for this run."
        )
        conn.close()
        return {}

    log.info("IDtools DB: %d URLs discovered", known)
    rows = conn.execute(
        "SELECT url, subject FROM urls WHERE subject IS NOT NULL AND subject != ''"
    ).fetchall()
    conn.close()

    index = {
        row[1].lower(): {"url": row[0], "subject": row[1]}
        for row in rows if row[1] and 3 < len(row[1]) < 100
    }
    log.info("IDtools index ready: %d subjects", len(index))
    return index


def _search_idtools(class_label: str, crop: str, idtools_index: dict[str, dict]) -> list[dict]:
    """Match class label against IDtools subjects; scrape visual section of best match."""
    if not idtools_index:
        return []

    import sys
    sys.path.insert(0, str(_WEB_SCRAPER_DIR))
    try:
        from scrape_idtools import scrape_factsheet
    except ImportError:
        return []

    terms = [t.lower() for t in (class_label + " " + crop).split() if len(t) > 3]
    if not terms:
        return []

    scored = sorted(
        ((sum(t in subj for t in terms), entry)
         for subj, entry in idtools_index.items()
         if any(t in subj for t in terms)),
        key=lambda x: x[0], reverse=True,
    )
    if not scored:
        return []

    sources = []
    for _score, match in scored[:2]:
        result = scrape_factsheet(match["url"])
        if result is None:
            continue
        _subject, content = result
        sources.append({
            "organization": "USDA APHIS IDtools",
            "title": match["subject"],
            "url": match["url"],
            "type": "fact_sheet",
            "accessed_date": str(date.today()),
            "excerpt": content[:EXCERPT_MAX],
            "search_query": f"{class_label} {crop}",
        })
        time.sleep(SLEEP)

    return sources


_CGIAR_DATASETS = [
    "CGIAR/ragas_gardian_evaluation_non_overlapping",
    "CGIAR/ragas_gardian_evaluation_overlapping",
]
_CGIAR_EMB_MODEL = "BAAI/bge-small-en-v1.5"
_CGIAR_EMB_CACHE = "cgiar_embeddings.npy"
_CGIAR_TEXT_CACHE = "cgiar_chunks.json"


def _fetch_cgiar_chunks() -> list[str]:
    try:
        import pyarrow.ipc as pa_ipc
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        log.warning("CGIAR disabled — missing dependency: %s", e)
        return []

    chunks: list[str] = []
    for ds_id in _CGIAR_DATASETS:
        try:
            path = hf_hub_download(
                repo_id=ds_id, filename="data-00000-of-00001.arrow", repo_type="dataset"
            )
            import pyarrow.ipc as pa_ipc
            reader = pa_ipc.open_stream(path)
            table = reader.read_all()
            ctx_col = table.column("contexts").to_pylist()
            before = len(chunks)
            for cell in ctx_col:
                if isinstance(cell, list):
                    chunks.extend(c for c in cell if c)
                elif cell:
                    chunks.append(cell)
            log.info("CGIAR %s: %d rows → %d chunks", ds_id, len(ctx_col), len(chunks) - before)
        except Exception as e:
            log.warning("CGIAR load '%s': %s", ds_id, e)
    return chunks


def _load_cgiar_index(cache_dir: Path) -> tuple | None:
    """
    Returns (model, embeddings, chunks) for RAG retrieval.

    First run: downloads chunks, encodes with bge-small-en-v1.5, saves to cache_dir.
    Subsequent runs: loads from cache in seconds.
    """
    import numpy as np
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.warning("sentence-transformers not installed — CGIAR RAG disabled")
        return None

    emb_path = cache_dir / _CGIAR_EMB_CACHE
    text_path = cache_dir / _CGIAR_TEXT_CACHE

    if emb_path.exists() and text_path.exists():
        log.info("Loading CGIAR embeddings from cache...")
        embeddings = np.load(str(emb_path))
        chunks = json.loads(text_path.read_text(encoding="utf-8"))
        log.info("CGIAR index ready: %d chunks, dim=%d", len(chunks), embeddings.shape[1])
        model = SentenceTransformer(_CGIAR_EMB_MODEL)
        return model, embeddings, chunks

    chunks = _fetch_cgiar_chunks()
    if not chunks:
        return None

    log.info("Encoding %d CGIAR chunks with %s (cached after this)...", len(chunks), _CGIAR_EMB_MODEL)
    model = SentenceTransformer(_CGIAR_EMB_MODEL)
    embeddings = model.encode(
        chunks, batch_size=512, show_progress_bar=True, normalize_embeddings=True
    )

    np.save(str(emb_path), embeddings)
    text_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    log.info("CGIAR index saved → %s", cache_dir)

    return model, embeddings, chunks


def _search_cgiar(
    class_label: str, crop: str, task_type: str,
    compressed_description: str,
    cgiar_index: tuple | None, top_k: int = 3,
) -> list[dict]:
    """Retrieve top-k CGIAR chunks by cosine similarity."""
    if cgiar_index is None:
        return []

    import numpy as np
    model, embeddings, chunks = cgiar_index

    if task_type in ("weed_identification", "crop_identification"):
        query = f"{compressed_description} visual appearance {class_label} plant identification"
    else:
        query = f"{compressed_description} visual symptoms {class_label} {crop} disease"

    q_emb = model.encode([query], normalize_embeddings=True)
    scores = (embeddings @ q_emb.T).flatten()
    top_idx = np.argsort(scores)[-top_k:][::-1]

    return [
        {
            "organization": "CGIAR GARDIAN",
            "title": "CGIAR GARDIAN Research Corpus",
            "url": "https://huggingface.co/datasets/CGIAR/ragas_gardian_evaluation_non_overlapping",
            "type": "research_document",
            "accessed_date": str(date.today()),
            "excerpt": chunks[i][:EXCERPT_MAX],
            "search_query": query,
            "score": round(float(scores[i]), 4),
        }
        for i in top_idx
        if scores[i] > 0.3
    ]


def _export_csv(jsonl_path: Path, csv_path: Path) -> list[dict]:
    """Read descriptions JSONL and write a flat CSV. Returns the rows."""
    rows: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            meta = record.get("metadata", {})
            rows.append({
                "agml_dataset":           meta.get("agml_dataset", ""),
                "crop":                   meta.get("crop", ""),
                "class_label":            meta.get("class_label", ""),
                "task_type":              meta.get("task_type", ""),
                "compressed_description": record["messages"][1]["content"],
            })
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["agml_dataset", "crop", "class_label", "task_type", "compressed_description"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} records → {csv_path}")
    return rows


def main(args: argparse.Namespace) -> None:
    dst = Path(args.output_db)
    dst.parent.mkdir(parents=True, exist_ok=True)

    csv_path = dst.parent / "descriptions_export.csv"
    rows = _export_csv(Path(args.descriptions_jsonl), csv_path)

    existing_db = json.loads(dst.read_text(encoding="utf-8")) if dst.exists() else {}
    existing_entries = existing_db.get("entries", {})

    seen: dict[str, dict] = {}
    for row in rows:
        key = _make_key(row["agml_dataset"], row["class_label"], row["crop"])
        if key not in seen:
            seen[key] = {k: row[k] for k in
                         ("agml_dataset", "crop", "class_label", "task_type", "compressed_description")}

    print("Loading USDA APHIS IDtools index...")
    idtools_index = _load_idtools_index()

    print("Loading CGIAR GARDIAN index (encoding on first run, cached after)...")
    cgiar_index = _load_cgiar_index(dst.parent)

    entries = dict(existing_entries)
    total = len(seen)

    for n, (key, meta) in enumerate(seen.items(), 1):
        if args.skip_existing and key in existing_entries:
            print(f"  {n}/{total} [skip] {meta['class_label']} ({meta['crop']})")
            continue

        if _is_healthy(meta["class_label"]):
            entries[key] = {**meta, "status": "healthy_skip", "sources": []}
            print(f"  {n}/{total} [healthy] {meta['class_label']} ({meta['crop']}) — skipped")
            save_json(dst, {"schema_version": "1.2", "built_date": str(date.today()), "entries": entries})
            continue

        sources = (
            _fetch_wikipedia(meta["class_label"], meta["crop"], meta["task_type"])
            + _search_idtools(meta["class_label"], meta["crop"], idtools_index)
            + _search_cgiar(
                meta["class_label"], meta["crop"], meta["task_type"],
                meta["compressed_description"], cgiar_index,
            )
        )
        status = "found" if any(s.get("excerpt") for s in sources) else "no_source_found"

        entries[key] = {**meta, "status": status, "sources": sources}
        print(f"  {n}/{total} {meta['class_label']} ({meta['crop']}) — {status} "
              f"[{len(sources)} sources]")

        save_json(dst, {"schema_version": "1.2", "built_date": str(date.today()), "entries": entries})

    found = sum(1 for e in entries.values() if e.get("status") == "found")
    print(f"\nDone. {found}/{len(entries)} entries have at least one source → {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build reference DB from Wikipedia, USDA APHIS IDtools, and CGIAR HuggingFace."
    )
    parser.add_argument(
        "--descriptions_jsonl", default="outputs/data_generation/agml_descriptions.jsonl",
        help="Path to agml_descriptions.jsonl produced by generate.py",
    )
    parser.add_argument(
        "--output_db", default="outputs/data_generation/reference_db.json",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip entries already present in an existing reference_db.json",
    )
    main(parser.parse_args())
