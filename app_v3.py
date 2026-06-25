# Standard library
import os
import math

# PDF Reader
import pdfplumber

# Anthropic
import anthropic

# HuggingFace
from sentence_transformers import SentenceTransformer

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

# 🤖── Claude calls ────────────────────────────────────────────────────────────
SYSTEM_CONTRACT = """You are a legal analyst specialising in contracts and terms of service.
Your job is to help users understand documents in plain, clear language.
Be precise, cite specific clauses when relevant, and only flag clauses that are
genuinely unusual or significantly disadvantageous compared to industry standards."""

def ask_claude(system: str, query: str) -> str:

    msgs = [{"role": "user", "content": query}]

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


