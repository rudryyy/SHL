#!/usr/bin/env python3
# src/api_fastapi.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import re

from src.retriever import load_index, search

APP_TITLE = "SHL GenAI Assessment Recommendation API"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"   # MUST match your built index
INDEX_DIR  = "index"

app = FastAPI(
    title=APP_TITLE,
    description="RAG API that recommends SHL assessments from the product catalog.",
    version="1.1.0",
)

# ----------------------------
# Load FAISS index + model
# ----------------------------
try:
    print(f"[INFO] Loading index '{INDEX_DIR}' with model '{MODEL_NAME}' ...")
    bundle = load_index(INDEX_DIR, model_name=MODEL_NAME)
    print("[READY] Index and model loaded.")
except Exception as e:
    bundle = None
    print(f"[WARN] Could not load index/model: {e}")

# ----------------------------
# Request schema
# ----------------------------
class QueryInput(BaseModel):
    query: str
    topk: int = 10   # we clamp to [1, 10]

# ----------------------------
# Utilities
# ----------------------------
def extract_duration_window(text: str):
    """
    Return (min_min, max_min) if query mentions duration like:
      - '40 minutes'
      - '1-2 hours'
      - 'about 60 min'
    Otherwise return None.
    """
    t = (text or "").lower()

    # Range, e.g. "1-2 hour(s)" or "45-60 min"
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*(hour|hr|hours|hrs|minute|min|minutes)\b", t)
    if m:
        a, b, unit = int(m.group(1)), int(m.group(2)), m.group(3)
        if "hour" in unit:
            a, b = a * 60, b * 60
        return (min(a, b), max(a, b))

    # Single value, e.g. "60 minutes", "1 hour"
    m = re.search(r"(\d+)\s*(hour|hr|hours|hrs|minute|min|minutes)\b", t)
    if m:
        v, unit = int(m.group(1)), m.group(2)
        if "hour" in unit:
            v = v * 60
        # soft window around target
        return (max(0, v - 15), v + 15)

    return None

def strong_terms_from_query(q: str) -> List[str]:
    # Extract words and keep a small whitelist of useful “strong” terms
    words = [w.lower() for w in re.findall(r"[a-zA-Z]+", q or "")]
    whitelist = {
        "python","sql","excel","powerbi","tableau","r","statistics","statistical",
        "developer","engineer","analyst","data","qa","testing","automation",
        "communication","stakeholder","manager","sales","marketing","java","javascript"
    }
    return [w for w in words if w in whitelist]

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "index_loaded": bundle is not None, "model": MODEL_NAME}

@app.get("/")
def root():
    return {"message": "Welcome to the SHL GenAI Assessment Recommendation API. See /docs for usage."}

@app.post("/recommend")
def recommend(inp: QueryInput) -> Dict[str, Any]:
    if bundle is None:
        raise HTTPException(status_code=503, detail="Index not loaded. Build index and restart API.")

    q = (inp.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query.")

    topk = max(1, min(10, int(inp.topk or 10)))

    # Retrieve a wider pool, then filter/rerank
    df = search(bundle, q, topk=topk * 4)

    # ---------- Keyword relevance gate (keeps on-topic items)
    strong = strong_terms_from_query(q)
    if strong:
        def on_topic(row):
            text = " ".join(str(row.get(c, "") or "") for c in ["title","description","tags","category"]).lower()
            return any(k in text for k in strong)
        df = df[df.apply(on_topic, axis=1)]

    # ---------- Duration awareness (soft boost if user asked for time)
    win = extract_duration_window(q)
    if win and "duration_min" in df.columns:
        lo, hi = win
        center = (lo + hi) / 2.0
        def dur_score(row):
            d = row.get("duration_min")
            if d is None or pd.isna(d):
                return 0.0
            try:
                d = float(d)
            except Exception:
                return 0.0
            # linear falloff around the center; tolerance ~= max(15, half the window)
            return max(0.0, 1.0 - abs(d - center) / max(15.0, (hi - lo) / 2.0))
        df["_dur_score"] = df.apply(dur_score, axis=1)
    else:
        df["_dur_score"] = 0.0

    # ---------- Final score = 0.85 * semantic + 0.15 * duration_fit
    if "similarity" in df.columns:
        df["_final"] = 0.85 * df["similarity"].astype(float) + 0.15 * df["_dur_score"].astype(float)
        df = df.sort_values("_final", ascending=False)

    # Keep top-K
    df = df.head(topk)

    # ---------- Build safe JSON
    recs: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        recs.append({
            "assessment_name": r.get("title", ""),
            "assessment_url":  r.get("url", ""),
            "test_type":       r.get("test_type", None),
            "level":           r.get("level", None),
            "language":        r.get("language", None),
            "duration_min":    r.get("duration_min", None),
            "similarity":      r.get("similarity", None),
            "description":     r.get("description", None),
        })

    return {"query": q, "count": len(recs), "recommendations": recs}
