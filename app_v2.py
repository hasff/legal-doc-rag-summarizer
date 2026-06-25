# Standard library
import os

# PDF Reader
import pdfplumber

# Anthropic
import anthropic

# Environment
from dotenv import load_dotenv
load_dotenv()


# ── Clients ──────────────────────────────────────────────────────────────────
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


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

    print("🍟 " * 40)
    print("                     PDF TEXT CHUNKS\n")
    for chunk_no, chunk in enumerate(pdf_text_chunks, start=1):
        print(f"👉 {chunk_no}) {chunk}\n")
    print("🍟 " * 40, '\n')    

    print(f"📦 Total chunks: {len(pdf_text_chunks)}")
    print(f"📏 Avg chunk size: {sum(len(c) for c in pdf_text_chunks) / len(pdf_text_chunks):.0f} chars\n")    

