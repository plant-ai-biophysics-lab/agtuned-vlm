#!/usr/bin/env python3
"""
audit_urls.py
=============
Quality-checks discovered IDtools URLs before running expensive inference.

Checks:
  1. URL reachability (HTTP status)
  2. Visual content score (keyword density)
  3. Subject name quality (catches stubs, navigation pages)
  4. Category distribution
  5. Duplicate subjects
  6. Sample of raw scraped text for manual review

Usage:
  python audit_urls.py                        # audit all pending URLs
  python audit_urls.py --sample 30            # spot-check 30 random URLs
  python audit_urls.py --fix                  # mark low-quality URLs as skip
  python audit_urls.py --report audit.json    # save full report to JSON
"""

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# ── Reuse scraping logic from main script ────────────────────────────────────
_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AgGemma-ResearchBot/1.0; "
        "academic data collection for agricultural AI research)"
    ),
    "Accept": "text/html,application/xhtml+xml",
}

_VISUAL_KEYWORDS = [
    "color",
    "colour",
    "symptom",
    "appear",
    "lesion",
    "spot",
    "pustule",
    "yellow",
    "brown",
    "white",
    "black",
    "orange",
    "red",
    "purple",
    "streak",
    "texture",
    "shape",
    "size",
    "necrosis",
    "chlorosis",
    "wilting",
    "mottl",
    "blight",
    "decay",
    "discolor",
    "pattern",
    "leaf",
    "stem",
    "root",
    "fruit",
    "flower",
    "margin",
    "surface",
    "wax",
    "frass",
    "damage",
    "mark",
    "ring",
    "band",
    "stripe",
    "powdery",
    "fuzzy",
    "waxy",
    "slimy",
    "crusty",
    "scaly",
]

_SKIP_HEADINGS = {
    "management",
    "control",
    "chemical",
    "pesticide",
    "treatment",
    "biology",
    "life cycle",
    "distribution",
    "economic",
    "references",
    "links",
    "contact",
    "navigation",
    "related",
    "legal",
    "disclaimer",
    "footer",
    "acknowledgment",
    "authors",
    "copyright",
    "citation",
    "glossary",
    "gallery",
    "key",
    "about",
}

_VISUAL_HEADINGS = {
    "symptom",
    "symptoms",
    "description",
    "identification",
    "visual",
    "appearance",
    "morphology",
    "diagnostic",
    "characteristics",
    "field identification",
    "signs",
    "damage",
    "recognition",
    "overview",
    "distinguishing features",
}

# Subjects that indicate a non-fact-sheet page got into the database
_STUB_SUBJECTS = {
    "about",
    "introduction",
    "overview",
    "authors",
    "acknowledgments",
    "references",
    "glossary",
    "gallery",
    "key",
    "contact",
    "copyright",
    "citation",
    "background",
    "scope",
    "home",
    "index",
    "navigation",
    "how to use",
    "lucid",
    "tabular",
    "morphology",
}


def visual_score(text: str) -> int:
    lower = text.lower()
    return sum(1 for kw in _VISUAL_KEYWORDS if kw in lower)


def is_stub_subject(subject: str) -> bool:
    s = subject.lower().strip()
    return any(stub in s for stub in _STUB_SUBJECTS) or len(s) < 4


def fetch_and_score(url: str, timeout: int = 15) -> Dict:
    """
    Fetch a URL and return quality metrics.
    """
    result = {
        "url": url,
        "http_status": None,
        "page_title": "",
        "visual_score": 0,
        "char_count": 0,
        "has_visual_section": False,
        "error": None,
        "text_sample": "",
    }

    try:
        resp = requests.get(url, headers=_HTTP_HEADERS, timeout=timeout)
        result["http_status"] = resp.status_code

        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code}"
            return result

    except requests.RequestException as e:
        result["error"] = str(e)
        return result

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(
        ["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]
    ):
        tag.decompose()

    # Page title
    h1 = soup.find("h1")
    if h1:
        result["page_title"] = h1.get_text(strip=True)

    # Check for visual headings
    for heading in soup.find_all(re.compile(r"^h[1-4]$")):
        h_text = heading.get_text(strip=True).lower()
        if any(vis in h_text for vis in _VISUAL_HEADINGS):
            result["has_visual_section"] = True
            break

    full_text = soup.get_text(separator=" ", strip=True)
    result["char_count"] = len(full_text)
    result["visual_score"] = visual_score(full_text)
    # First 400 chars of meaningful content for manual review
    result["text_sample"] = full_text[:400].strip()

    return result


def grade(r: Dict) -> str:
    """
    Assign a quality grade to a URL result.
      A  — strong visual content, good for QA generation
      B  — moderate content, likely usable
      C  — thin content, may produce weak QA pairs
      F  — not a real fact sheet (stub, error, no content)
    """
    if r["error"] or r["http_status"] != 200:
        return "F"
    if r["char_count"] < 200:
        return "F"
    if r["visual_score"] >= 8 and r["has_visual_section"]:
        return "A"
    if r["visual_score"] >= 4:
        return "B"
    if r["visual_score"] >= 1:
        return "C"
    return "F"


def run_audit(
    db_path: str,
    sample_n: Optional[int],
    fix: bool,
    report_path: Optional[str],
    delay: float = 0.8,
):

    conn = sqlite3.connect(db_path)

    rows = conn.execute(
        """SELECT url, tool, subject, category
           FROM urls
           WHERE status IN ('pending', 'scraped')
           ORDER BY RANDOM()"""
    ).fetchall()

    if not rows:
        print("No pending URLs found in database.")
        return

    if sample_n:
        rows = rows[:sample_n]
        print(
            f"\nAuditing {len(rows)} randomly sampled URLs (of {conn.execute('SELECT COUNT(*) FROM urls WHERE status IN ("pending","scraped")').fetchone()[0]} total pending) ...\n"
        )
    else:
        print(f"\nAuditing all {len(rows)} pending URLs ...\n")

    results = []
    grades = Counter()
    categories = Counter()
    subjects_seen = defaultdict(list)  # subject → list of urls (for dup detection)

    for i, (url, tool, subject, category) in enumerate(rows):
        # Flag stub subjects without fetching
        if is_stub_subject(subject):
            r = {
                "url": url,
                "tool": tool,
                "subject": subject,
                "category": category,
                "grade": "F",
                "reason": "stub subject name",
                "http_status": None,
                "visual_score": 0,
                "has_visual_section": False,
                "char_count": 0,
                "text_sample": "",
                "error": "stub",
            }
            results.append(r)
            grades["F"] += 1
            print(f"  [{i + 1}/{len(rows)}] F  {subject!r:<40} (stub subject)")
            continue

        metrics = fetch_and_score(url)
        g = grade(metrics)
        grades[g] += 1
        categories[category] += 1

        # Track for duplicate detection
        norm_subject = subject.lower().strip()
        subjects_seen[norm_subject].append(url)

        r = {
            "url": url,
            "tool": tool,
            "subject": subject,
            "category": category,
            "grade": g,
            "reason": "",
            **metrics,
        }
        results.append(r)

        status_str = (
            f"HTTP {metrics['http_status']}"
            if metrics["http_status"] != 200
            else f"score={metrics['visual_score']:2d} chars={metrics['char_count']:5d}"
        )
        print(f"  [{i + 1:3d}/{len(rows)}] {g}  {subject:<45} {status_str}")

        time.sleep(delay)

    # ── Duplicate detection ────────────────────────────────────────────────
    duplicates = {s: urls for s, urls in subjects_seen.items() if len(urls) > 1}

    # ── Apply fixes if requested ───────────────────────────────────────────
    skipped = 0
    if fix:
        for r in results:
            if r["grade"] == "F":
                conn.execute(
                    "UPDATE urls SET status='skip', error=? WHERE url=?",
                    (r.get("reason") or r.get("error") or "audit:F", r["url"]),
                )
                skipped += 1
        conn.commit()
        print(f"\n✓ Marked {skipped} URLs as 'skip' in database")

    # ── Print summary ──────────────────────────────────────────────────────
    total = len(results)
    print("\n" + "═" * 60)
    print("AUDIT SUMMARY")
    print("═" * 60)
    print(f"\nURLs audited : {total}")
    print(f"\nGrade distribution:")
    for g in ["A", "B", "C", "F"]:
        count = grades[g]
        pct = 100 * count / total if total else 0
        bar = "█" * int(pct / 2)
        label = {
            "A": "Strong visual content",
            "B": "Moderate content",
            "C": "Thin content",
            "F": "Stub / error / no content",
        }[g]
        print(f"  {g}  {bar:<25} {count:4d} ({pct:4.1f}%)  {label}")

    print(f"\nCategory distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat:<12} {count:4d}")

    if duplicates:
        print(f"\nDuplicate subjects ({len(duplicates)} found):")
        for subj, urls in list(duplicates.items())[:10]:
            print(f"  '{subj}'")
            for u in urls:
                print(f"    {u}")
    else:
        print(f"\nNo duplicate subjects found ✓")

    # ── Low scoring sample for manual review ──────────────────────────────
    c_grade = [r for r in results if r["grade"] == "C"]
    if c_grade:
        print(f"\nC-grade samples (thin content — review manually):")
        for r in c_grade[:5]:
            print(f"\n  Subject : {r['subject']}")
            print(f"  URL     : {r['url']}")
            print(f"  Score   : {r['visual_score']} | Chars: {r['char_count']}")
            print(f"  Sample  : {r.get('text_sample', '')[:200]}")

    f_grade_errors = [
        r for r in results if r["grade"] == "F" and r.get("error") != "stub"
    ]
    if f_grade_errors:
        print(f"\nF-grade errors (not stubs):")
        for r in f_grade_errors[:5]:
            print(f"  {r['subject']:<40} {r.get('error', '')}")

    print("\n" + "═" * 60)
    usable = grades["A"] + grades["B"]
    print(f"Estimated usable URLs: {usable}/{total} ({100 * usable / total:.0f}%)")
    if not fix and grades["F"] > 0:
        print(
            f"Tip: run with --fix to mark {grades['F']} F-grade URLs "
            f"as 'skip' so inference skips them."
        )
    print("═" * 60)

    # ── Save report ────────────────────────────────────────────────────────
    if report_path:
        report = {
            "total_audited": total,
            "grades": dict(grades),
            "categories": dict(categories),
            "duplicates": {k: v for k, v in list(duplicates.items())[:50]},
            "results": [
                {k: v for k, v in r.items() if k != "text_sample"} for r in results
            ],
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nFull report saved to: {report_path}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Audit IDtools URLs before running QA inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db",
        default="idtools_scraper.db",
        help="SQLite database (default: idtools_scraper.db)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Spot-check N random URLs instead of all",
    )
    parser.add_argument(
        "--fix", action="store_true", help="Mark F-grade URLs as 'skip' in the database"
    )
    parser.add_argument(
        "--report", default=None, help="Save full JSON report to this path"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.8,
        help="Seconds between requests (default 0.8)",
    )
    args = parser.parse_args()

    if not __import__("pathlib").Path(args.db).exists():
        print(f"Database not found: {args.db}")
        print("Run discovery first: bash run_scraper.sh discover")
        sys.exit(1)

    run_audit(
        db_path=args.db,
        sample_n=args.sample,
        fix=args.fix,
        report_path=args.report,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
