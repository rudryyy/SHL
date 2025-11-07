#!/usr/bin/env python3
"""
Retriever utility to load FAISS index and run queries.
"""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

@dataclass
class IndexBundle:
    index: faiss.Index
    vectors: np.ndarray
    meta: pd.DataFrame
    model: SentenceTransformer

def load_index(indexdir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> IndexBundle:
    index = faiss.read_index(str(Path(indexdir) / "faiss.index"))
    vectors = np.load(str(Path(indexdir) / "vectors.npy"))
    meta = pd.read_parquet(str(Path(indexdir) / "meta.parquet"))
    model = SentenceTransformer(model_name)
    return IndexBundle(index=index, vectors=vectors, meta=meta, model=model)

def search(bundle: IndexBundle, query: str, topk: int = 10) -> pd.DataFrame:
    q = bundle.model.encode([query], normalize_embeddings=True).astype(np.float32)
    D, I = bundle.index.search(q, topk)
    I, D = I[0], D[0]
    out = bundle.meta.iloc[I].copy()
    out["similarity"] = D
    return out.reset_index(drop=True)
