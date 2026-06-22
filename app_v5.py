# Standard library
import os
import math

# PDF Reader
import pdfplumber

# Anthropic
import anthropic

# HuggingFace
from sentence_transformers import SentenceTransformer

# BM25
from rank_bm25 import BM25Okapi

# Environment
from dotenv import load_dotenv
load_dotenv()


# ── Clients ──────────────────────────────────────────────────────────────────
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')


# 🚨 Models can change over time - the following are valid at the time of this writing Jun 2026
# CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_MODEL = "claude-haiku-4-5"

# ── PDF extraction ────────────────────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file) -> str:
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n\n".join(
            page.extract_text() or "" for page in pdf.pages
        )
    
# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else len(text)
    return [c for c in chunks if c.strip()]    

# ── Embeddings ────────────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    return embeddings_model.encode(texts).tolist()

def embed_query(query: str) -> list[float]:
    return embeddings_model.encode([query]).tolist()[0]

# ── Vector search (cosine) ────────────────────────────────────────────────────
def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

def vector_search(query_emb: list[float], embeddings: list[list[float]], k: int = 5) -> list[tuple[int, float]]:
    scores = [(i, cosine_similarity(query_emb, emb)) for i, emb in enumerate(embeddings)]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

# ── Tokenizing ────────────────────────────────────────────────────────────────
def tokenize_texts(texts: list[str]) -> list[list[str]]:
    return [c.lower().split() for c in texts]

def tokenize_query(query: str) -> list[str]:
    return query.lower().split()

# ── BM25 search ───────────────────────────────────────────────────────────────
def bm25_search(query_tokens: list[str], bm25: BM25Okapi, k: int = 5) -> list[tuple[int, float]]:
    scores = bm25.get_scores(query_tokens)
    return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────
def rrf_merge(
    vector_results: list[tuple[int, float]],
    bm25_results:   list[tuple[int, float]],
    k_rrf: int = 60,
    top_k: int = 5
) -> list[int]:
    scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(vector_results):
        scores[idx] = scores.get(idx, 0) + 1 / (k_rrf + rank + 1)
    for rank, (idx, _) in enumerate(bm25_results):
        scores[idx] = scores.get(idx, 0) + 1 / (k_rrf + rank + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:top_k]

# ── Hybrid retrieval ──────────────────────────────────────────────────────────
def hybrid_retrieve(query: str, chunks: list[str], embeddings: list[list[float]], bm25: BM25Okapi, top_k: int = 5) -> list[str]:
    query_emb    = embed_query(query)
    vec_results  = vector_search(query_emb, embeddings, k=top_k * 2)
    query_tokens = tokenize_query(query)
    bm25_results = bm25_search(query_tokens, bm25, k=top_k * 2)
    best_indices = rrf_merge(vec_results, bm25_results, top_k=top_k)
    return [chunks[i] for i in best_indices]


# 🤖── Claude calls ────────────────────────────────────────────────────────────
SYSTEM_CONTRACT = """You are a legal analyst specialising in contracts and terms of service.
Your job is to help users understand documents in plain, clear language.
Be precise, cite specific clauses when relevant, and only flag clauses that are
genuinely unusual or significantly disadvantageous compared to industry standards."""

def ask_claude(system: str, user: str) -> str:

    msgs = [{"role": "user", "content": user}]

    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=system,
        messages=msgs
    )

    return response.content[0].text    

# ─────────────────────────────────────────────
# 🚀 ENTRY POINT - TESTING 
# ─────────────────────────────────────────────
if __name__ == "__main__":

    from pathlib import Path
    PDFS_DIR = Path(__file__).parent / "tos_docs"

    file_path = PDFS_DIR / "danger_zone_rag_test.pdf"


    pdf_text = extract_text_from_pdf(file_path)
    pdf_text_chunks = chunk_text(pdf_text)

    question = f"""Hey claude can you explain to me whats up with the 'AI agent' info in the doc? 
    Also tell me in what parts the document it appears."""

    chunks_embeddings = embed_texts(pdf_text_chunks)   
    question_embeddings = embed_query(question)
    
    vector_search_result = vector_search(question_embeddings, chunks_embeddings)

    for chunk_idx, score in vector_search_result:
        print(f"🎯 {score} => {pdf_text_chunks[chunk_idx]}\n\n")


    chunks_tokens = tokenize_texts(pdf_text_chunks)
    bm25 = BM25Okapi(chunks_tokens) # indexing
    query_tokens = tokenize_query(question)

    bm25_search_result = bm25_search(query_tokens, bm25)
    for chunk_idx, score in bm25_search_result:
        print(f"🔍 {score} => {pdf_text_chunks[chunk_idx]}\n\n")

    print('🧐' * 50)

    hybrid_search_result = hybrid_retrieve(question, pdf_text_chunks, chunks_embeddings, bm25)
    for chunk_rank, chunk in enumerate(hybrid_search_result, start=1):
        print(f"🎯 + 🔍 {chunk_rank} => {chunk}\n\n")


