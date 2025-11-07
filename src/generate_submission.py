#!/usr/bin/env python3
import argparse, sys
import pandas as pd

# robust import whether run from repo root or src/
try:
    from src.retriever import load_index, search
except ModuleNotFoundError:
    import os, sys as _sys
    _sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from retriever import load_index, search  # type: ignore

def main(indexdir, model_name, test_csv, out_csv, topk):
    # load index/model
    bundle = load_index(indexdir, model_name=model_name)

    # read test queries; auto-detect the query column
    df = pd.read_csv(test_csv)
    qcol = None
    for c in df.columns:
        if str(c).strip().lower() in {"query","queries","jd","job_description","text"}:
            qcol = c
            break
    if qcol is None:
        raise ValueError(f"Could not find a query column in {test_csv}. "
                         "Expected one of: query, queries, jd, job_description, text")

    rows = []
    for q in df[qcol].astype(str).tolist():
        res = search(bundle, q, topk=topk)
        for url in res.get("url", []):
            rows.append({"Query": q, "Assessment_url": url})

    out = pd.DataFrame(rows, columns=["Query","Assessment_url"])
    out.to_csv(out_csv, index=False)
    print(f"[DONE] Wrote {len(out)} rows to {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indexdir", default="index")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--test", default="data/test_queries.csv")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    main(args.indexdir, args.model, args.test, args.out, args.topk)
