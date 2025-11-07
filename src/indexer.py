#!/usr/bin/env python3
"""
Builds the embedding index (FAISS) for SHL catalog.

Usage:
  python src/indexer.py --catalog data/shl_catalog.csv --outdir index \
    --model sentence-transformers/all-MiniLM-L6-v2
  # Tip: for better quality try:
  # --model sentence-transformers/all-mpnet-base-v2
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

DOC_TEMPLATE = (
    "Assessment Name: {title}. "
    "Category: {category}. "
    "Type: {test_type}. "
    "Level: {level}. "
    "Duration: {duration_min} minutes. "
    "Language: {language}. "
    "Tags: {tags}. "
    "Description: {description}. "
)

SAFE_COLS = [
    "assessment_id","title","url","description","category",
    "test_type","level","duration_min","language","tags"
]

def sg(row, key):
    """safe-get as str; returns '' if key missing or value is NaN."""
    if key not in row or pd.isna(row[key]):
        return ""
    return str(row[key])

def build_doc(row):
    return DOC_TEMPLATE.format(
        title=sg(row, "title"),
        category=sg(row, "category"),
        test_type=sg(row, "test_type"),
        level=sg(row, "level"),
        duration_min=sg(row, "duration_min"),
        language=sg(row, "language"),
        tags=sg(row, "tags"),
        description=sg(row, "description"),
    )

def main(catalog_path: str, outdir: str, model_name: str):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(catalog_path)
    if df.empty:
        raise SystemExit(f"Catalog is empty: {catalog_path}")

    # Ensure missing expected columns exist (fill with empty)
    for c in SAFE_COLS:
        if c not in df.columns:
            df[c] = ""

    # Build document text
    df["doc"] = df.apply(build_doc, axis=1)

    # Embed
    print(f"[INFO] Using model: {model_name}")
    model = SentenceTransformer(model_name)
    X = model.encode(df["doc"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    X = X.astype(np.float32)

    # FAISS index (cosine via inner product on normalized vectors)
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)

    # Save artifacts
    faiss.write_index(index, str(out / "faiss.index"))
    np.save(str(out / "vectors.npy"), X)
    df[SAFE_COLS].to_parquet(str(out / "meta.parquet"), index=False)
    print(f"[DONE] Saved index to {out} (n={len(df)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True, help="Path to data/shl_catalog.csv")
    ap.add_argument("--outdir", default="index", help="Output dir for FAISS + meta")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-Transformer model name")
    args = ap.parse_args()
    main(args.catalog, args.outdir, args.model)
