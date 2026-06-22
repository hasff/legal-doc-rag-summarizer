# Standard library
import os
import math
import re

# PDF Reader
import pdfplumber

# Streamlit
import streamlit as st

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
# 🖼️ Streamlit
# ─────────────────────────────────────────────
# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_pdfs(uploaded_files):
    all_chunks = []
    with st.status("Processing PDFs...", expanded=True) as status:
        for f in uploaded_files:
            st.write(f"📄 Extracting text from **{f.name}**...")
            text = extract_text_from_pdf(f)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            st.write(f"   → {len(chunks)} chunks created")

        st.write(f"🔢 Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = embed_texts(all_chunks)

        st.write("📚 Building BM25 index...")
        chunks_tokens = tokenize_texts(all_chunks)
        bm25 = BM25Okapi(chunks_tokens)

        status.update(label="✅ Ready!", state="complete")

    st.session_state.chunks     = all_chunks
    st.session_state.embeddings = embeddings
    st.session_state.bm25_index = bm25
    st.session_state.ready      = True
    st.session_state.messages   = []

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Legal Document Analyser",
    page_icon="⚖️",
    layout="wide"
)

# Init state
for key, default in [
    ("ready", False),
    ("chunks", []),
    ("embeddings", []),
    ("bm25_index", None),
    ("messages", []),
    ("danger", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ Legal Doc Analyser")
    st.caption("Upload legal document to analyse them.")
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        if st.button("🚀 Process Documents", use_container_width=True, type="primary"):
            preprocess_pdfs(uploaded_files)
            st.rerun()

    if st.session_state.ready:
        st.success(f"✅ {len(st.session_state.chunks)} chunks indexed")
        st.divider()

        # Danger Score
        st.subheader("🚨 Danger Score")
        if st.button("Analyse Risk", use_container_width=True):
            with st.spinner("Analysing..."):
                st.session_state.danger = compute_danger_score(st.session_state.chunks)

        if st.session_state.danger:
            d = st.session_state.danger
            score = d.get("score", 0)
            color = "🟢" if score <= 3 else "🟡" if score <= 6 else "🔴"
            st.metric("Risk Level", f"{color} {score}/10")
            st.caption(d.get("summary", ""))
            if d.get("red_flags"):
                st.subheader("⚠️ Red Flags")
                for flag in d["red_flags"]:
                    if isinstance(flag, dict):
                        st.warning(f"**{flag.get('clause', '')}**\n\n{flag.get('issue', '')}")
                    else:
                        st.warning(flag)

# ── Main area ─────────────────────────────────────────────────────────────────
if not st.session_state.ready:
    st.markdown("""
    ## How it works
    1. Upload one or more PDF documents via the sidebar
    2. Click **Process Documents** — text is extracted, chunked, and indexed
    3. Ask questions in the chat
    4. Use **Danger Score** to get an instant risk assessment
    5. Use **Simplify** to translate legalese into plain English
    """)
    st.stop()

tab_chat, tab_simplify = st.tabs(["💬 Chat", "🔍 Simplify Clause"])

# ── Chat tab ──────────────────────────────────────────────────────────────────
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything about the contract..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching and thinking..."):

                chunks      = st.session_state.chunks
                embeddings  = st.session_state.embeddings
                bm25        = st.session_state.bm25_index

                answer = answer_question(prompt, chunks, embeddings, bm25)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Simplify tab ──────────────────────────────────────────────────────────────
with tab_simplify:
    st.subheader("Paste a clause to simplify")
    clause_input = st.text_area(
        "Legal text",
        height=200,
        placeholder="Paste any clause or paragraph from the contract here..."
    )
    if st.button("✨ Simplify", type="primary") and clause_input.strip():
        with st.spinner("Translating legalese..."):

            chunks      = st.session_state.chunks
            embeddings  = st.session_state.embeddings
            bm25        = st.session_state.bm25_index

            simplified = simplify_clause(clause_input, chunks, embeddings, bm25)
        st.subheader("Plain English version")
        st.info(simplified)