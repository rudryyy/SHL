#!/usr/bin/env python3
# src/app_streamlit.py

import os
import sys
import streamlit as st
import pandas as pd
from typing import List

# --- Make imports work whether you run from project root or from src/ ---
try:
    # Works when you run: streamlit run src/app_streamlit.py (cwd = project root)
    from src.retriever import load_index, search
except ModuleNotFoundError:
    # Works when cwd becomes src/ (some Streamlit setups)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from retriever import load_index, search  # type: ignore

# ---------- Page config ----------
st.set_page_config(
    page_title="SHL GenAI Assessment Recommender (RAG)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Constants (must match your index build) ----------
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # <- keep in sync with indexer/evaluate/api
INDEX_DIR = "index"  # relative to project root

# ---------- Cached loader ----------
@st.cache_resource(show_spinner=True)
def get_bundle(model_name: str):
    return load_index(INDEX_DIR, model_name=model_name)

# ---------- Sidebar ----------
st.sidebar.header("Index")
st.sidebar.write(
    "Ensure you’ve built the index with:\n\n"
    "`python src/indexer.py --catalog data/shl_catalog.csv --outdir index "
    "--model sentence-transformers/all-mpnet-base-v2`"
)

try:
    bundle = get_bundle(MODEL_NAME)
    st.sidebar.success("Index loaded ✅")
except Exception as e:
    st.sidebar.error(f"Failed to load index:\n\n{e}")
    st.stop()

# ---------- Main UI ----------
st.title("SHL GenAI Assessment Recommender (RAG)")

q = st.text_area(
    "Enter a job description or natural-language query",
    height=160,
    placeholder=(
        "e.g., Hiring mid-level data analyst with skills in Excel, SQL, and statistics. "
        "Also wants strong stakeholder communication. 60 minutes"
    ),
)

topk = st.slider("Top-K results", 1, 10, 5)

def _select_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        "title", "url", "test_type", "level", "language",
        "duration_min", "similarity", "description"
    ]
    return [c for c in preferred if c in df.columns]

def _linkify(u: str) -> str:
    return f"[link]({u})" if isinstance(u, str) and u.strip() else ""

if st.button("Recommend", type="primary"):
    if not q.strip():
        st.warning("Please enter a query.")
        st.stop()

    with st.spinner("Searching catalog…"):
        try:
            res = search(bundle, q, topk=topk)

            cols = _select_columns(res)
            if not cols:
                st.error("No expected columns found in search results.")
                st.stop()

            view = res[cols].rename(columns={
                "title": "Assessment",
                "url": "URL",
                "test_type": "Type",
                "level": "Level",
                "language": "Language",
                "duration_min": "Duration (min)",
                "similarity": "Similarity",
                "description": "Description",
            })

            # Make URLs clickable
            if "URL" in view.columns:
                view["URL"] = view["URL"].apply(_linkify)

            st.subheader("Results")
            st.dataframe(view, use_container_width=True)

            # Download CSV
            csv = view.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="recommendations.csv",
                mime="text/csv",
            )

        except AssertionError:
            st.error(
                "Embedding dimension mismatch: your app and FAISS index are using different models.\n\n"
                "Fix:\n"
                "1) Ensure MODEL_NAME here is "
                "`sentence-transformers/all-mpnet-base-v2` (matches the index).\n"
                "2) Rebuild index if needed.\n"
                "3) Clear Streamlit cache then rerun:  `streamlit cache clear`"
            )
        except Exception as e:
            st.error(f"Search failed: {e}")
