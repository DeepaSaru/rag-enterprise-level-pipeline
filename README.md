# Enterprise RAG Assistant

This project builds a retrieval-augmented generation (RAG) assistant over UK visa guidance PDFs.

## Project Files

- `RAG.ipynb` builds a beginner-friendly RAG pipeline for one Standard Visitor visa PDF.
- `RAG-Enterprise-level-Pipeline.ipynb` extends the pipeline across all PDFs in `Data/`.
- `app.py` provides a Streamlit chat interface over the same document collection.
- `Data/` contains the source visa guidance PDFs.

## Setup

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

The app loads the PDFs, cleans extracted text, chunks the documents, builds a FAISS vector store with Hugging Face sentence embeddings, and answers questions with `google/flan-t5-base`.
