# Standard library
import os
import math
import re

# PDF Reader
import pdfplumber

# Anthropic
import anthropic

# HuggingFace
from sentence_transformers import SentenceTransformer

# BM25
from rank_bm25 import BM25Okapi

# Environment
import os
from dotenv import load_dotenv
load_dotenv()


# ── Clients ──────────────────────────────────────────────────────────────────
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')


# 🚨 Models can change over time - the following are valid at the time of this writtings Jun 2026
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

# ── BM25 search ───────────────────────────────────────────────────────────────
def bm25_search(query: str, bm25: BM25Okapi, k: int = 5) -> list[tuple[int, float]]:
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
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
    bm25_results = bm25_search(query, bm25, k=top_k * 2)
    best_indices = rrf_merge(vec_results, bm25_results, top_k=top_k)
    return [chunks[i] for i in best_indices]


# 🤖── Claude calls ────────────────────────────────────────────────────────────
SYSTEM_CONTRACT = """You are a legal analyst specialising in contracts and terms of service.
Your job is to help users understand documents in plain, clear language.
Be precise, cite specific clauses when relevant, and only flag clauses that are
genuinely unusual or significantly disadvantageous compared to industry standards."""

def ask_claude(system: str, user: str, prefill= False) -> str:

    msgs = [{"role": "user", "content": user}]

    # put words in claude's mouth
    # to force claude to return json since it "thinks" it already started writing json
    if prefill:
        msgs.append({"role": "assistant", "content": "{"})    

    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=system,
        messages=msgs
    )

    # at this point it just answer what was in fault
    # example: "score": 7, "summary": "...", "red_flags": [...]}
    # it will not include the openning of json! We must add it
    # "{" + '"score": 7, "summary": "...", "red_flags": [...]}'
    return  ("{" if prefill else "") + response.content[0].text  


# 🤖── Claude calls - actions ────────────────────────────────────────────────────
def compute_danger_score(chunks: list[str]) -> dict:
    sample = "\n\n---\n\n".join(chunks[:20])
    prompt = f"""Analyse these contract excerpts and return a JSON object with:
    - score: integer 1-10 (1=very safe, 10=extremely risky). Use the full range fairly:
    most standard commercial contracts should score between 3-5.
    Only score 7+ if there are clauses that are genuinely predatory or highly unusual.
    - summary: one sentence explaining the score
    - red_flags: list of up to 5 objects, each with exactly two keys:
    "clause" (short title) and "issue" (explanation).    
    Only include clauses that are genuinely concerning, not standard legal boilerplate.

    <excerpts>
    {sample}
    </excerpts>
    
    Return ONLY valid JSON, no markdown, no backticks, no explanation.
    """
    raw = ask_claude(SYSTEM_CONTRACT, prompt, True)
    import json
    try:
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(cleaned)
    except Exception:
        # Try extracting JSON object with regex as fallback
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {"score": 0, "summary": "Could not parse score.", "red_flags": []}


def rag_query(question: str, chunks: list[str], embeddings: list[list[float]], bm25: BM25Okapi, template_prompt: str, top_k= 5):
    context_chunks = hybrid_retrieve(question, chunks, embeddings, bm25, top_k)
    context = "\n\n---\n\n".join(context_chunks)
    prompt = template_prompt.format(context= context, question= question)

    return ask_claude(SYSTEM_CONTRACT, prompt)    

def answer_question(question: str, chunks: list[str], embeddings: list[list[float]], bm25: BM25Okapi) -> str:
    template_prompt = """Answer the user's question based exclusively on the contract excerpts below.
    If the answer is not in the excerpts, say so clearly.

    <contract_excerpts>
    {context}
    </contract_excerpts>

    <question>
    {question}
    </question>"""
    return rag_query(question, chunks, embeddings, bm25, template_prompt, top_k=3)

def simplify_clause(question: str, chunks: list[str], embeddings: list[list[float]], bm25: BM25Okapi) -> str:
    template_prompt = """Rewrite the following legal clause in plain, simple English.
    Use the related contract excerpts below for additional context if helpful.

    <related_context>
    {context}
    </related_context>

    <clause>
    {question}
    </clause>"""
    return rag_query(question, chunks, embeddings, bm25, template_prompt, top_k=5)

# ─────────────────────────────────────────────
# 🚀 ENTRY POINT - TESTING 
# ─────────────────────────────────────────────
def _test_compute_danger_score(pdf_text_chunks: list[str]):
    danger_score = compute_danger_score(pdf_text_chunks)
    
    print()
    print("✂️  " * 50)
    print(f"Score: {danger_score.get('score', 0)}")
    print(f"Summary: {danger_score.get('summary', 'None')} \n")    
    for rf in danger_score.get('red_flags', []):
        clause = rf.get('clause', 'None')
        issue = rf.get('issue', 'None')
        print(f"➡️  clause: {clause} \n➡️  issue: {issue} \n\n")


def _test_answer_question(question: str, chunks: list[str], chunks_embeddings: list[list[float]], bm25: BM25Okapi):
    result = answer_question(question, pdf_text_chunks, chunks_embeddings, bm25)

    print()
    print("✂️  " * 50)   
    print(" ===> answer_question")
    print(f"question: {question} \n") 
    print(f"answer: {result} \n\n") 


def _test_simplify_clause(clause: str, chunks: list[str], chunks_embeddings: list[list[float]], bm25: BM25Okapi):
    result = simplify_clause(clause, pdf_text_chunks, chunks_embeddings, bm25)

    print()
    print("✂️  " * 50)    
    print(" ===> simplify_clause")
    print(f"clause: {clause} \n") 
    print(f"answer: {result} \n\n") 


if __name__ == "__main__":

    from pathlib import Path
    PDFS_DIR = Path(__file__).parent / "tos_docs"

    file_path = PDFS_DIR / "Microsoft Services Agreement.pdf"
    file_path = PDFS_DIR / "google_terms_of_service_en_eu.pdf"
    file_path = PDFS_DIR / "danger_zone_rag_test.pdf"


    pdf_text = extract_text_from_pdf(file_path)
    pdf_text_chunks = chunk_text(pdf_text)

    chunks_embeddings = embed_texts(pdf_text_chunks)   
    tokenized = [c.lower().split() for c in pdf_text_chunks]
    bm25 = BM25Okapi(tokenized)



    # # 1)
    # _test_compute_danger_score(pdf_text_chunks)

    # 2)
    question = "What the document is about?"
    _test_answer_question(question, pdf_text_chunks, chunks_embeddings, bm25)

    # 3) 
    clause = """3.3 Real Estate Agent Obligations
Licensed real estate agents must act in the best interest of their client throughout the property
transaction lifecycle. Agents are prohibited from representing conflicting interests in the same
transaction without written disclosure and informed consent from both parties. Commission
structures must be disclosed prior to engagement (Disclosure Form: REA-DISC-2024). Agents must"""
    _test_simplify_clause(clause, pdf_text_chunks, chunks_embeddings, bm25)