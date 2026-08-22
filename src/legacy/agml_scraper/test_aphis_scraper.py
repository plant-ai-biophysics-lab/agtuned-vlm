"""
Quick test: fetch APHIS plant-pests-diseases pages and find the right CSS selectors.
Run this on the server before integrating into build_reference_db.py.

Usage:
    python test_aphis_scraper.py
"""

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (research; academic non-commercial use)"}
TIMEOUT = 15

SAMPLE_URLS = [
    "https://www.aphis.usda.gov/plant-pests-diseases/alb",
    "https://www.aphis.usda.gov/plant-pests-diseases/golden-nematode",
    "https://www.aphis.usda.gov/plant-pests-diseases/black-stem-rust-barberry",
]

# Candidate CSS selectors to try — we want the one with the main body text
SELECTORS = [
    "main",
    "article",
    ".field--name-body",
    ".field--type-text-with-summary",
    "#content",
    ".layout-container",
    ".usa-prose",
    "div.node__content",
    "div[class*='content']",
]

# Also test what a slug-based URL looks like for a known disease name
SLUG_TEST_CLASSES = [
    ("early-blight",   "https://www.aphis.usda.gov/plant-pests-diseases/early-blight"),
    ("late-blight",    "https://www.aphis.usda.gov/plant-pests-diseases/late-blight"),
    ("angular-leaf-spot", "https://www.aphis.usda.gov/plant-pests-diseases/angular-leaf-spot"),
]


def fetch_and_probe(url: str) -> None:
    print(f"\n{'='*70}")
    print(f"URL: {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        print(f"Status: {r.status_code}  |  Content-length: {len(r.text)}")
        if r.status_code != 200:
            print("  → skip")
            return

        soup = BeautifulSoup(r.text, "html.parser")
        print(f"Title: {soup.title.string.strip() if soup.title else 'N/A'}")

        print("\nSelector probe:")
        for sel in SELECTORS:
            el = soup.select_one(sel)
            if el:
                txt = el.get_text(separator=" ", strip=True)
                print(f"  [{sel}]  {len(txt)} chars  ✓")
                if len(txt) > 100:
                    print(f"    Preview: {txt[:400]}")
                    print()
            else:
                print(f"  [{sel}]  — not found")

        # Show all unique top-level div classes to spot the right container
        divs = soup.find_all("div", class_=True, limit=30)
        classes = set()
        for d in divs:
            for c in d.get("class", []):
                classes.add(c)
        print(f"\n  Top div classes: {sorted(classes)[:20]}")

    except Exception as e:
        print(f"  ERROR: {e}")


def test_slug_construction() -> None:
    print(f"\n{'='*70}")
    print("Slug construction test (disease name → URL):")
    for name, url in SLUG_TEST_CLASSES:
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            print(f"  {name:<30} → {r.status_code}")
        except Exception as e:
            print(f"  {name:<30} → ERROR: {e}")


if __name__ == "__main__":
    for url in SAMPLE_URLS:
        fetch_and_probe(url)

    test_slug_construction()

    print(f"\n{'='*70}")
    print("Next steps based on results:")
    print("  1. Note which CSS selector consistently has the body text")
    print("  2. Check if slug URLs (disease-name → 200 or 404) work for unknown classes")
    print("  3. If slugs give 404s, we fall back to site search")
