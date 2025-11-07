 SHL GenAI Assessment Recommendation (RAG)

This repo builds a RAG-style recommender over the SHL product catalog.

 Structure
SHL/
  data/
    train_queries.csv
    train_tidy_query_url.csv
    test_queries.csv
    shl_catalog.csv
  index/
    faiss.index
    vectors.npy
    meta.parquet
  src/
    shl_catalog_crawler.py
    indexer.py
    retriever.py
    evaluate.py
    api_fastapi.py
    app_streamlit.py

 Quickstart

 0) Install
pip install -r requirements.txt

 1) Build the index (from data/shl_catalog.csv)
python src/indexer.py --catalog data/shl_catalog.csv --outdir index

 2) Evaluate on train set (Mean Recall@10)
python src/evaluate.py --train data/train_tidy_query_url.csv --indexdir index

 3) Run API
uvicorn src.api_fastapi:app --host 0.0.0.0 --port 8000 --reload
# Health:      GET  http://localhost:8000/health
# Recommend:  POST  http://localhost:8000/recommend  {"query":"Looking for ... "}

 4) Run Streamlit UI
streamlit run src/app_streamlit.py
