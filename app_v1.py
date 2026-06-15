# Standard library
import os

# PDF Reader
import pdfplumber

# Anthropic
import anthropic

# Environment
import os
from dotenv import load_dotenv
load_dotenv()


# ── Clients ──────────────────────────────────────────────────────────────────
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# 🚨 Models can change over time - the following are valid at the time of this writtings Jun 2026
# CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_MODEL = "claude-haiku-4-5"

# ── PDF extraction ────────────────────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file) -> str:
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n\n".join(
            page.extract_text() or "" for page in pdf.pages
        )


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

    import time
    from pathlib import Path
    PDFS_DIR = Path(__file__).parent / "tos_docs"

    file_path = PDFS_DIR / "danger_zone_rag_test.pdf"

    __sep_size = 40

    t0 = time.time()
    pdf_text = extract_text_from_pdf(file_path)
    print(f"⏱️ PDF extracted in {time.time() - t0:.2f}s\n")

    # print("📰 " * __sep_size)
    # print("                     PDF TEXT\n")
    # print(pdf_text)
    # print("📰 " * __sep_size, '\n')

    question = f"""Hey claude can you explain to me whats up with the 'AI agent' info in the doc? 
    Also tell me in what parts the document it appears. 
    Here is the info: {pdf_text}"""
    print(f"😎 says:\n", question, '\n')

    t1 = time.time()
    answer = ask_claude(SYSTEM_CONTRACT, question)
    print(f"⏱️ Claude answered in {time.time() - t1:.2f}s\n")

    print(f"🤖 says:\n", answer)

