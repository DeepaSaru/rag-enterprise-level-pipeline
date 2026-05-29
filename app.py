import re
from pathlib import Path

from langchain.schema import Document
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "Data"
TOP_K_RESULTS = 5
CONTEXT_CHUNKS = 3
MAX_CONTEXT_CHARS = 1800


st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon=":mag:",
    layout="wide",
)


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\d{2}/\d{2}/\d{4},\s*\d{2}:\d{2}\s*Print.*?GOV\.UK", " ", text)
    text = re.sub(r"https://www\.gov\.uk/[^\s]+\s+\d+/\d+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@st.cache_resource(show_spinner="Loading PDFs and building the vector index...")
def build_retriever():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"PDF folder not found: {DATA_DIR}")

    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {DATA_DIR}")

    documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata["source"] = pdf_path.name
        documents.extend(loaded_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS}), len(pdf_paths), len(chunks)


@st.cache_resource(show_spinner="Loading the local answer model...")
def load_generator():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
    )


def format_sources(docs: list[Document]) -> list[str]:
    sources = []
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page")
        label = f"{source}, page {page + 1}" if isinstance(page, int) else source

        if label not in seen:
            sources.append(label)
            seen.add(label)

    return sources


def ask_question(query: str) -> dict[str, object]:
    retriever, _, _ = build_retriever()
    generator = load_generator()

    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs[:CONTEXT_CHUNKS])[:MAX_CONTEXT_CHARS]

    prompt = f"""
Answer the question using only the context below.
If the answer is not in the context, say: I could not find the answer in the provided documents.

Context:
{context}

Question: {query}

Answer:
"""

    response = generator(
        prompt,
        max_new_tokens=180,
        do_sample=False,
    )

    return {
        "answer": response[0]["generated_text"],
        "sources": format_sources(retrieved_docs),
        "retrieved_docs": retrieved_docs,
    }


st.title("Enterprise RAG Assistant")
st.caption("Ask questions across UK visa guidance documents.")

try:
    _, pdf_count, chunk_count = build_retriever()
    st.sidebar.success("Knowledge base ready")
    st.sidebar.write(f"PDFs indexed: {pdf_count}")
    st.sidebar.write(f"Chunks indexed: {chunk_count}")
except Exception as exc:
    st.error(f"Unable to build the knowledge base: {exc}")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


query = st.chat_input("Ask a question about the UK visa documents...")

if query:
    query = query.strip()
    if not query:
        st.stop()

    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and generating an answer..."):
            result = ask_question(query)

        st.markdown(result["answer"])

        if result["sources"]:
            st.markdown("**Sources**")
            for source in result["sources"]:
                st.markdown(f"- {source}")

        with st.expander("Retrieved evidence"):
            for index, doc in enumerate(result["retrieved_docs"], start=1):
                st.markdown(f"**Chunk {index}**")
                st.json(doc.metadata)
                st.write(doc.page_content)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
        }
    )
