"""
Stage 1b — Chunk all accessible CGIAR document datasets into passages.

Handles two schemas:
  - chapters-based: gardian-ai-ready-docs (title/abstract/chapters)
  - content-based:  gardian-cigi-ai-documents, cirad-ai-documents,
                    embrapa-ai-documents, usda-nal-ai-documents-en

Outputs:
  outputs/stage3_gardian_passages.jsonl
  outputs/stage3_gardian_stats.json

Usage:
  uv run python experiments/ag_finetune/data_prep/03_chunk_gardian.py
"""

import json
import re
from pathlib import Path

import huggingface_hub

OUT_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE    = 400   # words
CHUNK_OVERLAP = 50
MIN_CHUNK_LEN = 80    # words

# Datasets and which schema they use
DATASETS = [
    ("CGIAR/gardian-ai-ready-docs",       "chapters"),
    ("CGIAR/gardian-cigi-ai-documents",   "content"),
    ("CGIAR/cirad-ai-documents",          "content"),
    ("CGIAR/embrapa-ai-documents",        "content"),
    ("CGIAR/usda-nal-ai-documents-en",    "content"),
    # CGIAR/ifpri-ai-documents — no data files uploaded yet
]

BOILERPLATE_RE = re.compile(
    r"^(references|acknowledgements?|table of contents|\d+|https?://\S+)$",
    re.IGNORECASE,
)


def is_boilerplate(text: str) -> bool:
    return bool(BOILERPLATE_RE.match(text.strip()))


def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start : start + CHUNK_SIZE])
        if len(words[start : start + CHUNK_SIZE]) >= MIN_CHUNK_LEN and not is_boilerplate(chunk):
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def extract_texts_chapters(row: dict) -> list[str]:
    """Extract text from chapters-based schema (gardian-ai-ready-docs)."""
    texts = []
    abstract = row.get("abstract") or ""
    if abstract.strip():
        texts.append(abstract.strip())
    chapters = row.get("chapters") or []
    if isinstance(chapters, list):
        for ch in chapters:
            if isinstance(ch, dict):
                body = ch.get("body") or ch.get("text") or ch.get("content") or ""
            else:
                body = str(ch)
            if body.strip():
                texts.append(body.strip())
    return texts


def extract_texts_content(row: dict) -> list[str]:
    """Extract text from content-based schema (cigi/cirad/embrapa/usda)."""
    content = row.get("content") or ""
    if isinstance(content, list):
        content = " ".join(str(c) for c in content)
    return [content.strip()] if content.strip() else []


def extract_meta(row: dict, source: str) -> dict:
    meta = row.get("metadata") or {}
    return {
        "source":   source,
        "title":    row.get("title") or (meta.get("title") if isinstance(meta, dict) else ""),
        "keywords": row.get("keywords") or [],
        "url":      (meta.get("url") if isinstance(meta, dict) else "") or "",
    }


def download_dataset(name: str, token: str | None) -> str | None:
    """Download dataset to local HF cache and return local path. Returns None on failure."""
    print(f"  Downloading {name} to local cache...")
    try:
        local_path = huggingface_hub.snapshot_download(
            repo_id=name,
            repo_type="dataset",
            token=token,
        )
        print(f"  ✅ Downloaded → {local_path}")
        return local_path
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        return None


def process_dataset(name: str, schema: str, local_path: str, f, stats: dict) -> None:
    from datasets import load_dataset
    short_name = name.split("/")[1]
    print(f"\n  Loading {short_name} from local cache...")
    try:
        ds = load_dataset(local_path, split="train")
    except Exception as e:
        print(f"  ❌ Load failed: {e}")
        stats["skipped_datasets"].append(name)
        return

    doc_count = chunk_count = skip_count = 0

    for row in ds:
        doc_count += 1
        meta = extract_meta(row, short_name)

        if schema == "chapters":
            texts = extract_texts_chapters(row)
        else:
            texts = extract_texts_content(row)

        for text in texts:
            for i, chunk in enumerate(chunk_text(text)):
                out = {
                    "stage":    "rag_qa",
                    "source":   short_name,
                    "chunk_id": f"{short_name}_{doc_count}_{i}",
                    "passage":  chunk,
                    **meta,
                }
                f.write(json.dumps(out) + "\n")
                chunk_count += 1

        if not texts:
            skip_count += 1

        if doc_count % 5000 == 0:
            print(f"    {doc_count:,} docs → {chunk_count:,} chunks so far")

    print(f"  ✅ {short_name}: {doc_count:,} docs → {chunk_count:,} chunks ({skip_count} empty)")
    stats["by_dataset"][short_name] = {"docs": doc_count, "chunks": chunk_count, "skipped": skip_count}
    stats["total_docs"]   += doc_count
    stats["total_chunks"] += chunk_count


def main() -> None:
    token = None
    token_path = Path.home() / ".cache/huggingface/token"
    if token_path.exists():
        token = token_path.read_text().strip()
        huggingface_hub.login(token=token, add_to_git_credential=False)
        print("Logged in to HuggingFace.")

    out_path  = OUT_DIR / "stage3_gardian_passages.jsonl"
    stats_path = OUT_DIR / "stage3_gardian_stats.json"

    # Load existing stats to support per-dataset resume
    stats: dict = {"total_docs": 0, "total_chunks": 0, "by_dataset": {}, "skipped_datasets": []}
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text())
        except Exception:
            pass

    done_datasets = set(stats.get("by_dataset", {}).keys())
    if done_datasets:
        print(f"  Resuming: already done {done_datasets}")

    with open(out_path, "a") as f:  # append so completed datasets survive restarts
        for name, schema in DATASETS:
            short_name = name.split("/")[1]
            if short_name in done_datasets:
                print(f"  Skipping {short_name} (already done)")
                continue

            local_path = download_dataset(name, token)
            if local_path is None:
                stats["skipped_datasets"].append(name)
                stats_path.write_text(json.dumps(stats, indent=2))
                continue

            process_dataset(name, schema, local_path, f, stats)
            stats_path.write_text(json.dumps(stats, indent=2))  # checkpoint after each dataset

    print(f"\nDone. {stats['total_chunks']:,} passages from {stats['total_docs']:,} docs → {out_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
