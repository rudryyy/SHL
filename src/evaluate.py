#!/usr/bin/env python3
import argparse, re
import pandas as pd
from collections import defaultdict
from urllib.parse import urlparse
from retriever import load_index, search

def norm(u: str) -> str:
    if not isinstance(u, str): return ""
    u = u.strip().lower()
    u = re.sub(r"^https?://(www\.)?", "", u)
    up = urlparse("https://" + u)
    return f"{up.netloc}{up.path.rstrip('/')}"

def recall_at_k(pred_urls, true_urls, k=10):
    P = [norm(u) for u in pred_urls[:k]]
    T = {norm(u) for u in true_urls if norm(u)}
    hits = sum(1 for u in P if u in T)
    denom = max(1, len(T))
    return hits / denom

def main(train_tidy: str, indexdir: str, model_name: str, k: int = 10):
    df = pd.read_csv(train_tidy)
    if not {"query","relevant_url"}.issubset(df.columns):
        raise SystemExit("Train tidy must have columns: query, relevant_url")

    truth = defaultdict(list)
    for _, r in df.iterrows():
        q = str(r["query"]).strip()
        u = str(r["relevant_url"]).strip()
        if q and u:
            truth[q].append(u)

    bundle = load_index(indexdir, model_name=model_name)

    rows = []
    for q, urls in truth.items():
        preds = search(bundle, q, topk=10)
        r = recall_at_k(preds["url"].tolist(), urls, k=k)
        rows.append({"query": q, "n_truth": len(urls), "recall_at_10": r})

    res = pd.DataFrame(rows).sort_values("recall_at_10", ascending=True).reset_index(drop=True)
    print(res.to_string(index=False))
    print(f"\nMean Recall@{k}: {res['recall_at_10'].mean():.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="data/train_tidy_query_url.csv")
    ap.add_argument("--indexdir", default="index", help="Directory with faiss.index, vectors.npy, meta.parquet")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Encoder model (must match index)")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    main(args.train, args.indexdir, args.model, args.k)
