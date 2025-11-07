#!/usr/bin/env python3
"""
Augment shl_catalog.csv with any labeled URLs from train_tidy_query_url.csv
that are missing from the catalog. Parses title/description/duration/etc.

Usage:
  python src/augment_catalog.py \
    --catalog data/shl_catalog.csv \
    --train data/train_tidy_query_url.csv \
    --out data/shl_catalog.csv \
    --delay 0.6
"""

import argparse
import re
import time
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SHL-RAG/1.1"}

# -----------------------------
# URL normalization (for overlap checks)
# -----------------------------
def norm_url(u: str) -> str:
    if not isinstance(u, str):
        return ""
    u = u.strip().lower()
    u = re.sub(r"^https?://(www\.)?", "", u)
    up = urlparse("https://" + u)
    return f"{up.netloc}{up.path.rstrip('/')}"

# -----------------------------
# Duration parsing (minutes)
# -----------------------------
DUR_PATTERNS = [
    r"(\d{1,3})\s*-\s*(\d{1,3})\s*(minutes|min)\b",     # 45-60 minutes
    r"(\d{1,3})\s*-\s*(\d{1,3})\s*(hours|hour|hrs|hr)\b",# 1-2 hours
    r"(\d{1,3})\s*(minutes|min)\b",                     # 60 minutes
    r"(\d{1,2})\s*(hours|hour|hrs|hr)\b",               # 1 hour
]

def parse_duration_minutes(text: str):
    if not text:
        return None
    t = text.lower()
    for pat in DUR_PATTERNS:
        m = re.search(pat, t)
        if not m:
            continue
        # ranges
        if len(m.groups()) == 3:
            a, b, unit = int(m.group(1)), int(m.group(2)), m.group(3)
            if unit.startswith("hour"):
                a, b = a * 60, b * 60
            return int(round((a + b) / 2))
        # single minutes
        if len(m.groups()) == 2 and m.group(2).startswith("min"):
            return int(m.group(1))
        # single hours
        if len(m.groups()) == 2 and m.group(2).startswith("hour"):
            return int(m.group(1)) * 60
    return None

# -----------------------------
# Page parsing
# -----------------------------
def parse_product_page(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    # title
    title = None
    h = soup.find(["h1", "h2"])
    if h:
        title = h.get_text(strip=True)
    if not title:
        ogt = soup.find("meta", property="og:title")
        if ogt and ogt.get("content"):
            title = ogt["content"].strip()

    # description (prefer meta, fallback to first <p>)
    desc = None
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        desc = meta["content"].strip()
    if not desc:
        ogd = soup.find("meta", property="og:description")
        if ogd and ogd.get("content"):
            desc = ogd["content"].strip()
    if not desc:
        p = soup.find("p")
        if p:
            desc = p.get_text(" ", strip=True)

    # full text for heuristics
    body = soup.get_text(" ", strip=True)
    body_low = body.lower()

    # very light heuristic for test_type (K vs P)
    test_type = None
    if any(k in body_low for k in ["personality", "behavior", "behaviour", "interpersonal", "communication", "situational judgment", "sjq", "sjt"]):
        test_type = "P"
    if any(k in body_low for k in ["knowledge", "skill", "technical", "coding", "programming", "python", "java", "sql",
                                   "numerical", "verbal", "logical", "cognitive", "aptitude", "automata"]):
        # If any of these appear, prefer Knowledge
        test_type = "K"

    # duration
    duration_min = parse_duration_minutes(body)

    # level (basic heuristic)
    level = None
    for lvl in ["graduate", "entry", "junior", "mid", "senior", "manager", "lead"]:
        if re.search(rf"\b{lvl}\b", body_low):
            level = "Graduate" if lvl == "graduate" else lvl.capitalize()
            break

    # language (basic)
    language = "English" if "english" in body_low else None

    # tags/category we don’t have; keep empty strings
    return {
        "title": title,
        "url": url,
        "description": desc,
        "category": "",
        "test_type": test_type,
        "level": level,
        "duration_min": duration_min,
        "language": language,
        "tags": ""
    }

# -----------------------------
# Main
# -----------------------------
def main(catalog_csv: str, train_tidy_csv: str, out_csv: str, delay: float = 0.6):
    cat = pd.read_csv(catalog_csv)
    train = pd.read_csv(train_tidy_csv)

    # Normalize URL sets for overlap
    have_norm = set(cat["url"].dropna().map(norm_url))
    need_urls = []
    for u in train["relevant_url"].dropna().astype(str):
        if norm_url(u) not in have_norm:
            need_urls.append(u)

    print(f"Missing labeled URLs not in catalog: {len(need_urls)}")

    new_rows = []
    for u in need_urls:
        try:
            r = requests.get(u, headers=UA, timeout=30)
            r.raise_for_status()
            row = parse_product_page(r.text, u)

            # fallback title/desc if page sparse
            if not row.get("title"):
                row["title"] = urlparse(u).path.strip("/").replace("-", " ").title() or "SHL Assessment"
            if not row.get("description"):
                row["description"] = row["title"]

            # make a simple id from path
            path = urlparse(u).path.strip("/").lower()
            row["assessment_id"] = re.sub(r"[^a-z0-9]+", "-", path).strip("-") or "shl-item"

            new_rows.append(row)
            print("Added:", u)
            time.sleep(delay)
        except Exception as e:
            print("Skip:", u, "->", e)

    if new_rows:
        add = pd.DataFrame(new_rows)
        # Keep existing columns if present; otherwise allow new ones
        merged = pd.concat([cat, add], ignore_index=True)
        # Drop exact duplicate URLs
        merged = merged.drop_duplicates(subset=["url"]).reset_index(drop=True)
        merged.to_csv(out_csv, index=False)
        print(f"Saved augmented catalog -> {out_csv} (n={len(merged)})")
    else:
        print("No new rows to add. Catalog unchanged.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/shl_catalog.csv")
    ap.add_argument("--train",   default="data/train_tidy_query_url.csv")
    ap.add_argument("--out",     default="data/shl_catalog.csv")
    ap.add_argument("--delay",   type=float, default=0.6)
    args = ap.parse_args()
    main(args.catalog, args.train, args.out, args.delay)
