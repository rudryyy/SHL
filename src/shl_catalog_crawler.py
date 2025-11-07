#!/usr/bin/env python3
"""
SHL Catalog Crawler — Individual Test Solutions Only

- Crawls SHL's product catalog page and discovers product pages.
- Visits each product page and extracts useful fields.
- Excludes "Pre-packaged Job Solutions" (we only want individual tests).
- Outputs a CSV: data/shl_catalog.csv

Run:
  python src/shl_catalog_crawler.py --out data/shl_catalog.csv --delay 0.8

Notes:
- If SHL changes HTML structure, adjust selectors in parse_catalog_list/parse_product_page.
- Be polite: keep a small --delay between requests.
"""

import re
import time
import argparse
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SHL-RAG-Research/1.0; +https://example.com)"
}


def fetch(url: str, delay: float) -> str:
    time.sleep(max(0.0, delay))
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    return r.text


def is_prepackaged(name: str) -> bool:
    name_l = (name or "").lower()
    # Heuristics to skip bundles / job solutions
    return (
        "pre-packaged" in name_l
        or "prepackaged" in name_l
        or "job solution" in name_l
        or "solution:" in name_l  # some titles may include this
    )


def parse_catalog_list(html: str):
    """
    Parse the catalog grid/list to discover product links.
    The page uses many internal anchors; we filter to likely product paths.
    """
    soup = BeautifulSoup(html, "html.parser")
    items = []

    # Broadly capture anchors, then filter by path patterns
    for a in soup.select("a[href]"):
        href = a["href"].strip()
        if not href:
            continue

        # Make absolute URL
        abs_url = urljoin(BASE, href)

        # Filter to product-like paths
        cond_product = (
            "/products/" in abs_url
            or "/solutions/products/" in abs_url
        )

        if cond_product:
            title = a.get_text(strip=True) or ""
            # Very short/empty titles are often nav items – ignore them
            if len(title) < 3:
                continue

            items.append({"title": title, "url": abs_url})

    # Deduplicate by URL
    uniq = []
    seen = set()
    for it in items:
        if it["url"] not in seen:
            uniq.append(it)
            seen.add(it["url"])
    return uniq


def parse_product_page(html: str, url: str):
    """
    Extract fields from a product page. Conservative and resilient.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Title
    title_el = soup.find(["h1", "h2"])
    title = title_el.get_text(strip=True) if title_el else None

    # Description (meta description, or first paragraph fallback)
    desc = None
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        desc = meta["content"].strip()
    if not desc:
        p = soup.find("p")
        if p:
            desc = p.get_text(" ", strip=True)

    # Tags/badges if present
    tags = [t.get_text(strip=True) for t in soup.select(".tag, .badge, .label") if t.get_text(strip=True)]

    # Heuristics for type, duration, level, language
    body_text = soup.get_text(" ", strip=True).lower()

    # Test Type: K (knowledge/skills) vs P (personality/behavior)
    test_type = None
    if any(k in body_text for k in ["personality", "behavior", "behaviour", "openness", "conscientiousness"]):
        test_type = "P"
    elif any(k in body_text for k in ["knowledge", "skill", "technical", "coding", "numerical", "verbal", "logical", "cognitive", "aptitude"]):
        test_type = "K"

    # Duration (e.g., "35 minutes")
    duration_min = None
    m = re.search(r"(\d{1,3})\s*(minutes|min)\b", body_text)
    if m:
        try:
            duration_min = int(m.group(1))
        except Exception:
            duration_min = None

    # Language (naive)
    language = None
    if "english" in body_text:
        language = "English"

    # Level (naive scan)
    level = None
    for lvl in ["entry", "graduate", "junior", "mid", "senior", "manager", "lead"]:
        if lvl in body_text:
            level = lvl.capitalize()
            break

    # Category via breadcrumbs (if present)
    category = None
    bc = soup.select("nav.breadcrumb a, .breadcrumb a")
    if bc:
        bc_texts = [b.get_text(strip=True) for b in bc if b.get_text(strip=True)]
        if bc_texts:
            category = " > ".join(bc_texts)

    return {
        "title": title,
        "url": url,
        "description": desc,
        "category": category,
        "test_type": test_type,
        "level": level,
        "duration_min": duration_min,
        "language": language,
        "tags": ";".join(tags) if tags else None,
    }


def main(out_csv: str, delay: float):
    print(f"[INFO] Crawling catalog: {CATALOG_URL}")
    catalog_html = fetch(CATALOG_URL, delay)
    candidates = parse_catalog_list(catalog_html)
    print(f"[INFO] Found {len(candidates)} candidate links before filtering/dedup")

    records = []
    seen_urls = set()

    for it in candidates:
        url = it["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)

        try:
            product_html = fetch(url, delay)
            pr = parse_product_page(product_html, url)
            title = pr.get("title") or it["title"]

            # Skip pre-packaged / job solutions
            if not title or is_prepackaged(title):
                continue

            pr["title"] = title

            # Generate a simple ID from the URL path
            path = urlparse(url).path.strip("/").lower()
            pr["assessment_id"] = re.sub(r"[^a-z0-9]+", "-", path).strip("-")

            records.append(pr)
            print(f"[OK] {title[:90]}")

        except requests.HTTPError as e:
            print(f"[HTTP] {url} -> {e}")
        except Exception as e:
            print(f"[ERR] {url} -> {e}")

    df = pd.DataFrame.from_records(records)
    if df.empty:
        print("[WARN] No products parsed. You may need to adjust selectors.")
    else:
        # Deduplicate and reorder cols
        df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
        cols = [
            "assessment_id", "title", "url", "description", "category",
            "test_type", "level", "duration_min", "language", "tags"
        ]
        df = df.reindex(columns=cols)

    # Ensure folder exists
    pd.Path = None
    from pathlib import Path
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[DONE] Saved {len(df)} assessments -> {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/shl_catalog.csv",
                        help="Output CSV path (default: data/shl_catalog.csv)")
    parser.add_argument("--delay", type=float, default=0.8,
                        help="Seconds to sleep between HTTP requests (default: 0.8)")
    args = parser.parse_args()
    main(args.out, args.delay)
