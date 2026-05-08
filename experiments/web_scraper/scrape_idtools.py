#!/usr/bin/env python3
"""
IDtools Agricultural Visual Descriptor QA Generator
====================================================
Scrapes USDA APHIS IDtools fact sheets and generates visual-descriptor
question-answer pairs using Qwen3.6-35B-A3B via the vLLM Python API.

Features:
  - Resumable: SQLite tracks every URL — stop and restart without losing work
  - Deduplication: already-processed URLs are skipped automatically
  - Memory monitoring: GPU + RAM logged each batch, warnings at thresholds
  - HuggingFace/vLLM inference-compatible output (JSONL, messages format)
  - Optional Unsloth SFT conversion via --convert-sft flag
  - Thinking mode disabled by default (faster, cheaper for extraction tasks)

Usage:
  python scrape_idtools.py [options]

Examples:
  # Discover URLs only (no inference)
  python scrape_idtools.py --discover-only

  # Full run with defaults
  python scrape_idtools.py --output idtools_qa.jsonl

  # Resume after interruption (safe to run again anytime)
  python scrape_idtools.py --output idtools_qa.jsonl

  # If 80GB fills up, try limiting context
  python scrape_idtools.py --max-model-len 2048 --gpu-mem 0.88
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import psutil
import requests
from bs4 import BeautifulSoup


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging(level: str, log_file: str = "scraper.log") -> logging.Logger:
    fmt = "%(asctime)s │ %(levelname)-8s │ %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(format=fmt, datefmt="%H:%M:%S", handlers=handlers)
    logger = logging.getLogger("idtools")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


log = logging.getLogger("idtools")


# ─────────────────────────────────────────────────────────────────────────────
# Memory monitoring
# ─────────────────────────────────────────────────────────────────────────────
def memory_stats() -> Dict:
    """Return current RAM and GPU memory stats."""
    ram = psutil.virtual_memory()
    stats = {
        "ram_used_gb":  round(ram.used  / 1e9, 2),
        "ram_total_gb": round(ram.total / 1e9, 2),
        "ram_pct":      round(ram.percent, 1),
    }
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(dev)
            reserved  = torch.cuda.memory_reserved(dev)
            total     = torch.cuda.get_device_properties(dev).total_memory
            stats.update({
                "gpu_allocated_gb": round(allocated / 1e9, 2),
                "gpu_reserved_gb":  round(reserved  / 1e9, 2),
                "gpu_total_gb":     round(total     / 1e9, 2),
                "gpu_pct":          round(100 * reserved / total, 1),
            })
    except Exception:
        pass
    return stats


def log_memory(label: str = "",
               warn_gpu_pct: float = 92.0,
               warn_ram_pct: float = 88.0):
    s = memory_stats()
    tag = f"[{label}] " if label else ""

    gpu_str = ""
    if "gpu_total_gb" in s:
        gpu_str = (
            f" | GPU {s['gpu_allocated_gb']}/{s['gpu_total_gb']} GB "
            f"reserved {s['gpu_reserved_gb']} GB ({s['gpu_pct']}%)"
        )

    log.info(
        f"{tag}RAM {s['ram_used_gb']}/{s['ram_total_gb']} GB "
        f"({s['ram_pct']}%){gpu_str}"
    )

    if s.get("gpu_pct", 0) > warn_gpu_pct:
        log.warning(
            f"⚠️  GPU memory at {s.get('gpu_pct')}% — above {warn_gpu_pct}% threshold. "
            "Consider reducing --batch-size or --max-model-len."
        )
    if s["ram_pct"] > warn_ram_pct:
        log.warning(
            f"⚠️  RAM at {s['ram_pct']}% — above {warn_ram_pct}% threshold. "
            "Risk of CPU-side OOM."
        )


# ─────────────────────────────────────────────────────────────────────────────
# SQLite checkpoint database
# ─────────────────────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS urls (
    url             TEXT PRIMARY KEY,
    tool            TEXT,
    subject         TEXT,
    category        TEXT,
    status          TEXT    DEFAULT 'pending',
    scraped_at      TEXT,
    processed_at    TEXT,
    error           TEXT
);

CREATE TABLE IF NOT EXISTS qa_pairs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT    NOT NULL,
    question    TEXT    NOT NULL,
    answer      TEXT    NOT NULL,
    subject     TEXT,
    category    TEXT,
    created_at  TEXT    DEFAULT (datetime('now')),
    FOREIGN KEY (url) REFERENCES urls(url)
);

CREATE INDEX IF NOT EXISTS idx_urls_status  ON urls(status);
CREATE INDEX IF NOT EXISTS idx_qa_url       ON qa_pairs(url);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.executescript(_SCHEMA)
    conn.commit()
    log.info(f"Database ready: {db_path}")
    return conn


def url_is_known(conn: sqlite3.Connection, url: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM urls WHERE url = ?", (url,)
    ).fetchone() is not None


def upsert_url(conn: sqlite3.Connection,
               url: str, tool: str, subject: str, category: str):
    conn.execute(
        """INSERT OR IGNORE INTO urls (url, tool, subject, category, status)
           VALUES (?, ?, ?, ?, 'pending')""",
        (url, tool, subject, category),
    )
    conn.commit()


def mark_url(conn: sqlite3.Connection,
             url: str, status: str, error: str = None):
    ts = datetime.utcnow().isoformat()
    if status == "scraped":
        conn.execute(
            "UPDATE urls SET status=?, scraped_at=? WHERE url=?",
            (status, ts, url),
        )
    else:
        conn.execute(
            "UPDATE urls SET status=?, processed_at=?, error=? WHERE url=?",
            (status, ts, error, url),
        )
    conn.commit()


def insert_qa_pairs(conn: sqlite3.Connection,
                    url: str, pairs: List[Dict],
                    subject: str, category: str):
    conn.executemany(
        """INSERT INTO qa_pairs (url, question, answer, subject, category)
           VALUES (?, ?, ?, ?, ?)""",
        [(url, p["question"], p["answer"], subject, category)
         for p in pairs],
    )
    conn.commit()


def get_pending_urls(conn: sqlite3.Connection) -> List[Dict]:
    """Return all pending/scraped URLs in random order (random = better diversity)."""
    rows = conn.execute(
        """SELECT url, tool, subject, category FROM urls
           WHERE status IN ('pending', 'scraped')
           ORDER BY RANDOM()"""
    ).fetchall()
    return [
        {"url": r[0], "tool": r[1], "subject": r[2], "category": r[3]}
        for r in rows
    ]


def db_summary(conn: sqlite3.Connection) -> str:
    stats = conn.execute(
        """SELECT status, COUNT(*) FROM urls GROUP BY status"""
    ).fetchall()
    qa_count = conn.execute("SELECT COUNT(*) FROM qa_pairs").fetchone()[0]
    stat_str = " | ".join(f"{s}: {c}" for s, c in stats)
    return f"URLs [{stat_str}] | QA pairs: {qa_count}"


# ─────────────────────────────────────────────────────────────────────────────
# IDtools URL discovery
# ─────────────────────────────────────────────────────────────────────────────
# IDtools actual URL structure (confirmed):
#   Tool index  : https://idtools.org/{tool_name}/index.cfm
#   Fact sheet listing: https://idtools.org/{tool_name}/index.cfm?packageID={id}
#   Individual fact sheet: https://idtools.org/{tool_name}/index.cfm?packageID={id}&entityID={id}
#   Alternate tools: https://idtools.org/tools/{number}/index.cfm
#
# Strategy:
#   1. Seed with known tool index pages
#   2. Crawler finds packageID links (fact sheet listings) → queues them
#   3. Crawler finds entityID links (individual fact sheets) → saves them

# Known tool index pages — all confirmed real subdirectories on idtools.org
# Include both tool index pages AND known package index pages directly
# so we don't depend on the crawler finding packageID links from redirected homepages
SEED_URLS = [
    # ── Package index pages (fact sheet listings) ─────────────────────────
    # These directly list all entityID fact sheet links in static HTML
    # citrus pests (61 fact sheets)
    "https://idtools.org/citrus_pests/index.cfm?packageID=63",
    # citrus diseases
    "https://idtools.org/citrus_diseases/index.cfm?packageID=14",
    # Phytophthora
    "https://idtools.org/phytophthora/index.cfm?packageID=1131",

    # ── Tool index pages ──────────────────────────────────────────────────
    # Crawler will find packageID links from navigation on these pages
    "https://idtools.org/citrus_pests/index.cfm",
    "https://idtools.org/citrus_diseases/index.cfm",
    "https://idtools.org/phytophthora/index.cfm",
    "https://idtools.org/aphid/index.cfm",
    "https://idtools.org/thrips/index.cfm",
    "https://idtools.org/tortricid/index.cfm",
    "https://idtools.org/sawfly/index.cfm",
    "https://idtools.org/diprion/index.cfm",
    "https://idtools.org/bbgus/index.cfm",
    "https://idtools.org/thebeemd/index.cfm",

    # ── Numeric tool IDs (alternate URL format) ───────────────────────────
    "https://idtools.org/tools/5/index.cfm",     # citrus diseases alt URL
    "https://idtools.org/tools/15/index.cfm",    # citrus pests alt URL
    "https://idtools.org/tools/1076/index.cfm",  # citrus ID edition 2

    # ── Broad portal — crawler will find more tool links here ─────────────
    "https://idtools.org/identify.cfm",
]

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AgGemma-ResearchBot/1.0; "
        "academic data collection for agricultural AI research)"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


def _parse_qs(url: str) -> Dict:
    """Parse query string of a URL into a dict."""
    from urllib.parse import parse_qs
    qs = urlparse(url).query
    return {k: v[0] for k, v in parse_qs(qs).items()}


def is_factsheet_url(url: str) -> bool:
    """
    A URL is a fact sheet if it has an entityID parameter.
    e.g. .../index.cfm?packageID=63&entityID=375
    """
    params = _parse_qs(url)
    return "entityID" in params


def is_package_index_url(url: str) -> bool:
    """
    A package index lists all fact sheets for a tool section.
    e.g. .../index.cfm?packageID=63  (no entityID)
    """
    params = _parse_qs(url)
    return "packageID" in params and "entityID" not in params


def is_tool_index_url(url: str) -> bool:
    """
    A tool index page — e.g. .../citrus_pests/index.cfm (no params, or pageID only)
    """
    parsed = urlparse(url)
    return (
        "idtools.org" in parsed.netloc
        and parsed.path.endswith("index.cfm")
        and "entityID" not in parsed.query
        and "packageID" not in parsed.query
    )


def infer_tool_name(url: str) -> str:
    parts = [p for p in urlparse(url).path.strip("/").split("/") if p]
    # Use the subdirectory name as tool name (e.g. 'citrus_pests')
    return parts[0].replace("_", " ").title() if parts else "unknown"


def infer_subject(url: str, link_text: str) -> str:
    text = (link_text or "").strip()
    # Use link text if it looks like a species/disease name
    if text and 3 < len(text) < 100 and not text.lower().startswith("http"):
        return text
    # Fall back to entityID
    params = _parse_qs(url)
    return f"Entity {params.get('entityID', '?')}"


def infer_category(url: str, tool_name: str = "") -> str:
    combined = (url + " " + tool_name).lower()
    if any(w in combined for w in ["disease", "pathogen", "phytophthora",
                                    "blight", "rot", "rust", "mold", "mildew",
                                    "fungal", "bacterial", "viral"]):
        return "disease"
    if any(w in combined for w in ["weed", "grass", "broadleaf", "invasive",
                                    "palm", "plant_id"]):
        return "weed"
    if any(w in combined for w in ["pest", "insect", "mite", "thrip", "fly",
                                    "bug", "beetle", "moth", "aphid", "scale",
                                    "mealybug", "nematode", "sawfly", "bee",
                                    "wasp", "psyllid", "whitefly"]):
        return "pest"
    return "unknown"


def discover_factsheet_urls(conn: sqlite3.Connection,
                             max_discover: int = 1500,
                             delay: float = 1.5) -> int:
    """
    Crawl IDtools from seed pages to find individual fact sheet URLs.

    Two-level crawl:
      Level 1: tool index pages (e.g. /citrus_pests/index.cfm)
               → find packageID links → queue them
      Level 2: package index pages (e.g. ?packageID=63)
               → find entityID links → save as fact sheets
    """
    crawled   = set()
    queue     = list(SEED_URLS)
    new_count = 0

    known = set(
        r[0] for r in conn.execute("SELECT url FROM urls").fetchall()
    )

    log.info(f"Starting URL discovery (max {max_discover}) ...")
    log.info(f"  Seed URLs: {len(queue)}")

    while queue and new_count < max_discover:
        url = queue.pop(0)
        url = url.split("#")[0]

        if url in crawled:
            continue
        crawled.add(url)

        try:
            resp = requests.get(url, headers=_HTTP_HEADERS, timeout=20)
        except requests.RequestException as e:
            log.debug(f"Request failed [{url}]: {e}")
            continue

        if resp.status_code != 200:
            log.debug(f"HTTP {resp.status_code} [{url}]")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("#", "mailto:", "javascript:")):
                continue

            abs_url = urljoin(base, href).split("#")[0]
            parsed  = urlparse(abs_url)

            # Stay within idtools.org
            if "idtools.org" not in parsed.netloc:
                continue
            if abs_url in crawled:
                continue

            link_text = a.get_text(strip=True)

            # ── Fact sheet (has entityID) → save it ──────────────────────
            if is_factsheet_url(abs_url):
                if abs_url not in known:
                    tool     = infer_tool_name(abs_url)
                    subject  = infer_subject(abs_url, link_text)
                    category = infer_category(abs_url, tool)
                    upsert_url(conn, abs_url, tool, subject, category)
                    known.add(abs_url)
                    new_count += 1
                    if new_count % 50 == 0:
                        log.info(f"  Found {new_count} fact sheets ...")

            # ── Package index (has packageID, no entityID) → crawl it ────
            elif is_package_index_url(abs_url):
                if abs_url not in crawled:
                    queue.append(abs_url)

            # ── Tool index (index.cfm, no params) → crawl it ─────────────
            elif is_tool_index_url(abs_url):
                if abs_url not in crawled:
                    queue.append(abs_url)

            # ── Any other idtools.org .cfm page → crawl it ───────────────
            # Be aggressive: queue all .cfm URLs so we don't miss
            # package index pages linked from redirected tool homepages
            elif (
                "idtools.org" in parsed.netloc
                and parsed.path.endswith(".cfm")
                and abs_url not in crawled
                and "entityID" not in parsed.query   # already handled above
            ):
                queue.append(abs_url)

        time.sleep(delay)

    log.info(f"Discovery finished — {new_count} new fact sheet URLs added")
    return new_count


# ─────────────────────────────────────────────────────────────────────────────
# Fact sheet scraping
# ─────────────────────────────────────────────────────────────────────────────
# Section headings that contain visual / identification content
_VISUAL_HEADINGS = {
    "symptom", "symptoms", "description", "identification",
    "visual", "appearance", "morphology", "morphological",
    "diagnostic", "characteristics", "field identification",
    "how to identify", "signs", "damage symptoms",
    "distinguishing features", "recognition", "overview",
}

# Section headings to skip entirely
_SKIP_HEADINGS = {
    "management", "control", "chemical control", "biological control",
    "pesticide", "treatment", "biology", "life cycle", "lifecycle",
    "distribution", "economic", "references", "links", "contact",
    "navigation", "related", "legal", "disclaimer", "footer",
    "similar species management", "prevention",
}

# Minimum visual keyword count to accept a page
_VISUAL_KEYWORDS = [
    "color", "colour", "symptom", "appear", "lesion", "spot", "pustule",
    "yellow", "brown", "white", "black", "orange", "red", "purple",
    "streak", "texture", "shape", "size", "necrosis", "chlorosis",
    "wilting", "mottl", "blight", "decay", "discolor", "pattern",
    "leaf", "stem", "root", "fruit", "flower", "margin", "surface",
]


def scrape_factsheet(url: str, timeout: int = 20) -> Optional[Tuple[str, str]]:
    """
    Fetch and extract visual content from a fact sheet page.
    Returns (subject_name, cleaned_visual_text) or None if insufficient content.
    """
    try:
        resp = requests.get(url, headers=_HTTP_HEADERS, timeout=timeout)
    except requests.RequestException as e:
        log.debug(f"Request error [{url}]: {e}")
        return None

    if resp.status_code != 200:
        log.debug(f"HTTP {resp.status_code} [{url}]")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Strip boilerplate
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "form", "button", "noscript", "iframe"]):
        tag.decompose()

    # Extract subject name
    subject = ""
    for tag in ["h1", "title"]:
        node = soup.find(tag)
        if node:
            subject = node.get_text(strip=True)
            break
    subject = re.sub(r"\s*[-|:]\s*IDtools.*$", "", subject, flags=re.IGNORECASE).strip()

    # Walk headings and collect visual sections
    visual_parts = []
    headings = soup.find_all(re.compile(r"^h[1-4]$"))

    for heading in headings:
        h_text = heading.get_text(strip=True).lower().rstrip(":")
        h_clean = re.sub(r"[^a-z ]", "", h_text).strip()

        if any(skip in h_clean for skip in _SKIP_HEADINGS):
            continue

        is_visual = any(vis in h_clean for vis in _VISUAL_HEADINGS)

        if not is_visual:
            continue

        # Collect paragraph siblings until next heading
        section_parts = [heading.get_text(strip=True) + ":"]
        for sib in heading.find_next_siblings():
            if sib.name and re.match(r"^h[1-4]$", sib.name):
                break
            text = sib.get_text(separator=" ", strip=True)
            if text and len(text) > 15:
                section_parts.append(text)

        if len(section_parts) > 1:
            visual_parts.append("\n".join(section_parts))

    # Fallback: use full page text if no structured sections found
    if not visual_parts:
        full = soup.get_text(separator="\n", strip=True)
        visual_parts = [full[:5000]]

    combined = "\n\n".join(visual_parts)

    # Sanity check — require minimum visual vocabulary
    lower = combined.lower()
    score = sum(1 for kw in _VISUAL_KEYWORDS if kw in lower)
    if score < 3:
        log.debug(f"Skipping {url} — visual score={score} (too low)")
        return None

    # Trim to safe context length (avoid overfilling model context)
    if len(combined) > 6000:
        combined = combined[:6000] + "\n[truncated]"

    return subject, combined.strip()


# ─────────────────────────────────────────────────────────────────────────────
# vLLM model + QA generation
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are an expert agricultural pathologist, entomologist, and botanist.
Your task is to generate question-answer pairs focused EXCLUSIVELY on the
VISUAL APPEARANCE and field IDENTIFICATION of plant diseases, pests, and weeds.

Rules:
1. Questions must ask what something LOOKS LIKE — colors, textures, shapes,
   sizes, patterns, lesion appearance, affected plant parts, distinctive markings.
2. Answers must describe visual characteristics in specific, concrete detail.
3. Do NOT generate questions about management, control, pesticides, biology,
   life cycles, economic impact, or distribution.
4. Return ONLY a valid JSON array — no prose, no markdown, no explanation.\
"""

_USER_TEMPLATE = """\
Read the following agricultural fact sheet content and generate exactly {n} \
question-answer pairs about the VISUAL APPEARANCE of the subject.

Good question types:
- "What does [X] look like on [crop]?"
- "What are the visual symptoms of [X]?"
- "How can I identify [X] by sight in the field?"
- "What color/shape/texture are the lesions caused by [X]?"
- "Which plant parts does [X] affect and what do they look like?"
- "What distinguishes [X] visually from similar conditions?"

Answers must be specific: mention colors, sizes, patterns, affected plant parts.

Fact sheet content:
---
{content}
---

Return ONLY a JSON array:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]"""


def load_model(model_name: str,
               gpu_memory_utilization: float = 0.90,
               max_model_len: int = 4096,
               quantization: Optional[str] = None):
    """
    Load Qwen3.6-35B-A3B via vLLM Python API.
    Returns (llm, tokenizer, sampling_params).

    Memory note:
      Qwen3.6-35B-A3B: 35B total params, 3B active (MoE).
      Full bf16 weights ≈ 70 GB. On an 80 GB A100 with
      gpu_memory_utilization=0.90 (72 GB), this is very tight.
      If you hit OOM:
        1. Lower --max-model-len to 2048
        2. Lower --gpu-mem to 0.88
        3. Or use a quantized model variant
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    log.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )

    log_memory("before model load")

    log.info(
        f"Loading vLLM model ...\n"
        f"  model               : {model_name}\n"
        f"  gpu_memory_util     : {gpu_memory_utilization}\n"
        f"  max_model_len       : {max_model_len}\n"
        f"  quantization        : {quantization or 'none (bfloat16)'}\n"
        f"  tensor_parallel_size: 1"
    )

    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype="bfloat16",
        quantization=quantization,         # None = full bf16
        trust_remote_code=True,
        enforce_eager=False,               # set True only if hitting CUDA graph OOM
        enable_prefix_caching=True,        # caches system prompt tokens across batch
        max_num_seqs=16,                   # limit concurrent sequences to control mem
    )

    sampling_params = SamplingParams(
        temperature=0.6,          # low but not greedy — some diversity in phrasing
        top_p=0.92,
        repetition_penalty=1.05,  # discourage repetitive answers
        max_tokens=1024,
        stop=["\n\n\n", "```"],
    )

    log_memory("after model load")
    return llm, tokenizer, sampling_params


def build_prompt(tokenizer, content: str, n_pairs: int) -> str:
    """
    Format the inference prompt using Qwen3's chat template.
    Thinking mode is disabled via enable_thinking=False.
    Falls back gracefully if the tokenizer doesn't support the kwarg.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _USER_TEMPLATE.format(
            content=content, n=n_pairs
        )},
    ]
    try:
        # Qwen3 series: enable_thinking=False disables <think> reasoning blocks
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Fallback for tokenizer versions that don't support enable_thinking
        log.debug("enable_thinking kwarg not supported — using standard template")
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def parse_qa_output(raw: str, max_pairs: int = 5) -> List[Dict]:
    """
    Parse the model's JSON output into a list of {question, answer} dicts.
    Robust to markdown fences, leading text, and partial JSON.
    """
    # Strip any accidental markdown code fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Find the first complete [...] block
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        log.debug(f"No JSON array found. Raw output (first 300 chars):\n{raw[:300]}")
        return []

    try:
        pairs = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        log.debug(f"JSON parse error: {e}\nRaw: {raw[:300]}")
        return []

    valid = []
    for item in pairs:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer",   "")).strip()
        # Basic quality filters
        if len(q) < 15 or len(a) < 30:
            continue
        if not any(kw in a.lower() for kw in ["color", "colour", "appear",
                                                "symptom", "lesion", "yellow",
                                                "brown", "white", "look", "spot",
                                                "shape", "texture", "red", "black"]):
            continue  # Reject answers with no visual vocabulary
        valid.append({"question": q, "answer": a})
        if len(valid) >= max_pairs:
            break

    return valid


def run_inference_batch(llm,
                        tokenizer,
                        sampling_params,
                        batch: List[Dict],
                        max_pairs: int = 5) -> List[Dict]:
    """
    Generate QA pairs for a batch of scraped fact sheets.
    Returns list of result dicts with url, subject, category, pairs.
    """
    prompts = [
        build_prompt(tokenizer, entry["content"], n_pairs=max_pairs)
        for entry in batch
    ]

    try:
        outputs = llm.generate(prompts, sampling_params)
    except Exception as e:
        log.error(f"vLLM inference error: {e}")
        return [
            {"url": e["url"], "subject": e["subject"],
             "category": e["category"], "pairs": [], "error": str(e)}
            for e in batch
        ]

    results = []
    for entry, output in zip(batch, outputs):
        raw   = output.outputs[0].text
        pairs = parse_qa_output(raw, max_pairs=max_pairs)
        results.append({
            "url":      entry["url"],
            "subject":  entry["subject"],
            "category": entry["category"],
            "pairs":    pairs,
            "error":    None,
        })
        log.debug(
            f"  {entry['subject']}: {len(pairs)} pairs "
            f"(tokens used: {len(output.outputs[0].token_ids)})"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Output writer
# ─────────────────────────────────────────────────────────────────────────────
def write_jsonl_inference(output_path: str, results: List[Dict]):
    """
    Append QA pairs to JSONL in HuggingFace/vLLM messages format.
    Each line is one complete Q&A exchange, ready for inference or SFT.

    Licensing note:
      IDtools content is produced by USDA APHIS PPQ in collaboration with
      university partners. As a US government work it is in the public domain
      under 17 U.S.C. § 105 (works of the US federal government are not
      subject to copyright protection). Generated QA pairs derived from this
      content inherit the same public domain status.
      Citation guidance: acknowledge USDA APHIS ITP and the specific tool.

    Output format:
    {
      "messages": [
        {"role": "user",      "content": "<question>"},
        {"role": "assistant", "content": "<answer>"}
      ],
      "metadata": {
        "source_name":    "USDA APHIS Identification Technology Program (ITP)",
        "source_url":     "<fact sheet URL>",
        "tool":           "<tool subdirectory, e.g. citrus_pests>",
        "subject":        "<species or disease name>",
        "category":       "disease | pest | weed | unknown",
        "task":           "visual_descriptor_qa",
        "license":        "Public Domain (17 U.S.C. § 105 — US government work)",
        "license_url":    "https://www.usa.gov/government-works",
        "attribution":    "USDA APHIS Plant Protection and Quarantine, Identification Technology Program. https://idtools.org",
        "generated_by":   "Qwen/Qwen3.6-35B-A3B via vLLM (QA pair synthesis)",
        "dataset_name":   "AgGemma-IDtools-VisualQA",
        "created_at":     "<ISO timestamp>"
      }
    }
    """
    ts = datetime.utcnow().isoformat() + "Z"
    with open(output_path, "a", encoding="utf-8") as f:
        for result in results:
            # Parse tool name from URL path
            tool_dir = ""
            try:
                path_parts = [p for p in urlparse(result["url"]).path.split("/") if p]
                tool_dir = path_parts[0] if path_parts else ""
            except Exception:
                pass

            for pair in result["pairs"]:
                record = {
                    "messages": [
                        {"role": "user",      "content": pair["question"]},
                        {"role": "assistant", "content": pair["answer"]},
                    ],
                    "metadata": {
                        # Provenance
                        "source_name":  "USDA APHIS Identification Technology Program (ITP)",
                        "source_url":   result["url"],
                        "tool":         tool_dir,
                        "subject":      result["subject"],
                        "category":     result["category"],
                        "task":         "visual_descriptor_qa",
                        # Licensing
                        "license":      "Public Domain (17 U.S.C. § 105 — US government work)",
                        "license_url":  "https://www.usa.gov/government-works",
                        "attribution":  (
                            "USDA APHIS Plant Protection and Quarantine, "
                            "Identification Technology Program (ITP). "
                            "https://idtools.org"
                        ),
                        # Generation
                        "generated_by": "Qwen/Qwen3.6-35B-A3B via vLLM (QA synthesis)",
                        "dataset_name": "AgGemma-IDtools-VisualQA",
                        "created_at":   ts,
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def convert_to_sft_format(input_path: str, output_path: str):
    """
    Convert inference JSONL (messages format) to Unsloth SFT conversations format.

    Unsloth SFT format:
    {
      "conversations": [
        {"from": "human", "value": "<question>"},
        {"from": "gpt",   "value": "<answer>"}
      ]
    }
    """
    written = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            if len(messages) < 2:
                continue
            sft_record = {
                "conversations": [
                    {"from": "human", "value": messages[0]["content"]},
                    {"from": "gpt",   "value": messages[1]["content"]},
                ],
                "metadata": record.get("metadata", {}),
            }
            fout.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
            written += 1
    log.info(f"SFT conversion complete: {written} records written to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate agricultural visual QA pairs from USDA IDtools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Core options
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B",
        help="HuggingFace model ID (default: Qwen/Qwen3.6-35B-A3B)")
    parser.add_argument("--db", default="idtools_scraper.db",
        help="SQLite checkpoint database (default: idtools_scraper.db)")
    parser.add_argument("--output", default="idtools_qa_dataset.jsonl",
        help="Output JSONL file (default: idtools_qa_dataset.jsonl)")

    # QA generation options
    parser.add_argument("--max-pairs", type=int, default=5,
        help="Max QA pairs per fact sheet, default 5")
    parser.add_argument("--batch-size", type=int, default=8,
        help="Inference batch size, default 8. Reduce if GPU OOM.")

    # vLLM options
    parser.add_argument("--max-model-len", type=int, default=4096,
        help="Max context length (default 4096). Reduce to 2048 if OOM.")
    parser.add_argument("--gpu-mem", type=float, default=0.90,
        help="GPU memory utilization fraction (default 0.90).")
    parser.add_argument("--quantization", default=None,
        choices=["awq", "gptq", "bitsandbytes"],
        help="vLLM quantization. Only use if model variant is pre-quantized.")

    # Scraping options
    parser.add_argument("--scrape-delay", type=float, default=1.5,
        help="Seconds between HTTP requests (default 1.5). Be polite.")
    parser.add_argument("--max-discover", type=int, default=1500,
        help="Max fact sheet URLs to discover (default 1500).")

    # Mode flags
    parser.add_argument("--discover-only", action="store_true",
        help="Discover URLs only, do not run inference.")
    parser.add_argument("--convert-sft", action="store_true",
        help="After generation, also write an Unsloth SFT-format JSONL.")

    # Misc
    parser.add_argument("--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity (default INFO).")
    parser.add_argument("--log-file", default="scraper.log",
        help="Log file path (default scraper.log).")

    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    log = setup_logging(args.log_level, args.log_file)

    log.info("=" * 64)
    log.info("IDtools Agricultural Visual QA Generator")
    log.info("=" * 64)
    log.info(f"  Model          : {args.model}")
    log.info(f"  Database       : {args.db}")
    log.info(f"  Output         : {args.output}")
    log.info(f"  Max pairs/sheet: {args.max_pairs}")
    log.info(f"  Batch size     : {args.batch_size}")
    log.info(f"  GPU mem util   : {args.gpu_mem}")
    log.info(f"  Max model len  : {args.max_model_len}")
    log.info("=" * 64)

    conn = init_db(args.db)
    log.info(f"DB state: {db_summary(conn)}")

    # ── URL discovery ─────────────────────────────────────────────────────────
    known_count = conn.execute("SELECT COUNT(*) FROM urls").fetchone()[0]

    if known_count < 50:
        log.info("Running URL discovery from IDtools ...")
        n_found = discover_factsheet_urls(
            conn,
            max_discover=args.max_discover,
            delay=args.scrape_delay,
        )
        log.info(f"Discovery: {n_found} new URLs added")
    else:
        log.info(
            f"Skipping discovery — {known_count} URLs already in DB. "
            f"Delete {args.db} to force re-discovery."
        )

    if args.discover_only:
        log.info(f"--discover-only: done. {db_summary(conn)}")
        return

    # ── Check for pending work ────────────────────────────────────────────────
    pending = get_pending_urls(conn)
    log.info(f"{len(pending)} fact sheets pending")

    if not pending:
        log.info("All URLs already processed. Nothing to do.")
        if args.convert_sft and Path(args.output).exists():
            sft_path = args.output.replace(".jsonl", "_sft.jsonl")
            convert_to_sft_format(args.output, sft_path)
        return

    # ── Load vLLM model ───────────────────────────────────────────────────────
    llm, tokenizer, sampling_params = load_model(
        model_name=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        quantization=args.quantization,
    )

    # ── Main scrape + generate loop ───────────────────────────────────────────
    total_pairs = 0
    total_done  = 0
    batch_buf   = []     # list of {url, subject, category, content}

    def flush_batch():
        """Run inference on the current batch buffer and save results."""
        nonlocal total_pairs, total_done

        if not batch_buf:
            return

        log.info(f"Running inference: batch of {len(batch_buf)} fact sheets ...")
        log_memory("pre-inference")

        results = run_inference_batch(
            llm, tokenizer, sampling_params,
            batch_buf,
            max_pairs=args.max_pairs,
        )

        log_memory("post-inference")

        for res in results:
            if res["error"]:
                mark_url(conn, res["url"], "failed", error=res["error"])
                log.warning(f"  ✗ {res['subject']}: inference error")
                continue

            if res["pairs"]:
                insert_qa_pairs(
                    conn, res["url"], res["pairs"],
                    res["subject"], res["category"]
                )
                write_jsonl_inference(args.output, [res])
                total_pairs += len(res["pairs"])
                total_done  += 1
                mark_url(conn, res["url"], "processed")
                log.info(
                    f"  ✓ {res['subject']} ({res['category']}): "
                    f"{len(res['pairs'])} pairs"
                )
            else:
                mark_url(conn, res["url"], "failed",
                         error="no valid pairs generated")
                log.warning(f"  ✗ {res['subject']}: model returned no valid pairs")

        batch_buf.clear()

    for i, entry in enumerate(pending):
        url      = entry["url"]
        subject  = entry["subject"]
        category = entry["category"]

        log.info(
            f"[{i+1}/{len(pending)}] Scraping: {subject} "
            f"({category}) ..."
        )

        result = scrape_factsheet(url, timeout=20)
        time.sleep(args.scrape_delay)

        if result is None:
            mark_url(conn, url, "no_content")
            log.info(f"  → Skipped (no visual content)")
            continue

        page_subject, content = result
        # Prefer scraped title over inferred subject if available
        if page_subject and len(page_subject) > 3:
            subject = page_subject
            conn.execute(
                "UPDATE urls SET subject=? WHERE url=?", (subject, url)
            )
            conn.commit()

        mark_url(conn, url, "scraped")

        batch_buf.append({
            "url":      url,
            "subject":  subject,
            "category": category,
            "content":  content,
        })

        is_last = (i == len(pending) - 1)
        if len(batch_buf) >= args.batch_size or is_last:
            flush_batch()
            log.info(
                f"Running total: {total_done} sheets done, "
                f"{total_pairs} QA pairs | {db_summary(conn)}"
            )

    # ── Finalize ──────────────────────────────────────────────────────────────
    log.info("=" * 64)
    log.info("COMPLETE")
    log.info(f"  Fact sheets processed : {total_done}")
    log.info(f"  QA pairs generated    : {total_pairs}")
    log.info(f"  Output                : {args.output}")
    log.info(f"  Database              : {args.db}")
    log_memory("final")

    if args.convert_sft and Path(args.output).exists():
        sft_path = args.output.replace(".jsonl", "_sft.jsonl")
        log.info(f"Converting to Unsloth SFT format → {sft_path}")
        convert_to_sft_format(args.output, sft_path)

    log.info("=" * 64)


if __name__ == "__main__":
    main()
