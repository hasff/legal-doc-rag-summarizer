# legal-doc-rag-summarizer

> A RAG pipeline that reads legal PDFs, answers questions about them, scores their risk, and simplifies legalese into plain English — built with Python, HuggingFace embeddings, BM25, and the Claude API.

> 💾 If this project looks useful, starring it now means you won't lose it later.

🗓️ **Status: June 2026**

---
## Picture this

You're about to register to an app just to see if it suits your needs, you just want to try it out...

You fill the forms and click next... then it complains about the *impossible* **CAPTCHA**...
you squint at the screen, try three times... finally you did it! Click next...

Now it complains because you didn't check the *"I have read and agree to the Terms of Service."*
You check it in a hurry and smash that next button. **FINALLY!!! 🥇🥇🥇**

Did you read it? Those terms you just "said" you did?

Of course not. No one reads that. After all, it is probably the biggest lie on the internet,
repeated billions of times every day.

But the devil is in the details. Somewhere in that wall of legal text there might be a clause
that lets the company sell your data, terminate your account without notice, or opt you in to
arbitration instead of a court. **You just agreed to it.**

With this tool you upload a PDF, and a few seconds later you have a risk score, a list of
red flags, a chat interface where you can ask anything about the document, and a way to paste
any clause and get it back in plain, everyday language anyone can understand (or even in any other
language you ask for).

![Legal Doc Analyser in action](assets/intro/screenshot_intro.jpg)

---

⚠️ **Heads up**

This is a personal learning project — not an official Anthropic resource.
It may contain errors, simplifications, or opinionated choices made for clarity over correctness.
Think of it as a **hands-on RAG tutorial**: each part builds on the previous one, so you always know why the next step exists.

Before you dive in, keep a few things in mind:

1. **Fast-Paced AI Evolution:** The AI landscape moves fast. Specific libraries or model names may change, but the RAG concepts taught here will stay relevant.
2. **Not production-ready:** This project was built to learn and teach. It has not been tested or hardened for production use.
3. **Built with AI Assistance:** This README was written with AI help, mainly for English refinement. The architecture, curriculum, and all technical decisions are my own.

---

# Key Concepts Demonstrated

✅ PDF text extraction
<br>✅ Text chunking strategies
<br>✅ HuggingFace sentence embeddings
<br>✅ Vector search with cosine similarity
<br>✅ BM25 lexical search
<br>✅ Hybrid retrieval with Reciprocal Rank Fusion (RRF)
<br>✅ RAG pipeline with Claude (Haiku)
<br>✅ Assistant prefill for structured JSON output
<br>✅ Streamlit UI

<a name="table-of-contents_"></a>

---

## Table of Contents

- [What is RAG?](#what-is-rag_)
- [Project Architecture](#project-architecture_)
- [Requirements](#requirements_)
- [Setup](#setup_)
- [Project Structure](#project-structure_)
- [Part 01 - The Naive Approach: Sending the Whole PDF to Claude](#part-1)
- [Part 02 - Divide and Conquer: Chunking the Document](#part-2)
- [Part 03 - Finding What Matters: Vector Search with Embeddings](#part-3)
- [Part 04 - Exact Match: BM25 Lexical Search](#part-4)
- [Part 05 - Best of Both Worlds: Hybrid Retrieval with RRF](#part-5)
- [Part 06 - How Risky Is This Contract? The Danger Score](#part-6)
- [Part 07 - Answering Questions and Simplifying Clauses: RAG in Action](#part-7)
- [Part 08 - Wrapping It All Up: The Streamlit Interface](#part-8)
- [Next Steps & Resources](#next-steps--resources_)
- [Get in Touch](#get-in-touch_)


<a name="what-is-rag_"></a>

---

## What is RAG?

#### ⚡ Quick Navigation: [⬅️ Table of Contents](#table-of-contents_) | [Project Architecture ➡️](#project-architecture_)

_TODO_

[↑ Back to Table of Contents](#table-of-contents_)

<a name="project-architecture_"></a>

---

## Project Architecture

#### ⚡ Quick Navigation: [⬅️ What is RAG?](#what-is-rag_) | [Requirements ➡️](#requirements_)

_TODO_

[↑ Back to Table of Contents](#table-of-contents_)

<a name="requirements_"></a>

---

## Requirements

#### ⚡ Quick Navigation: [⬅️ Project Architecture](#project-architecture_) | [Setup ➡️](#setup_)

- Python 3.10+
- An Anthropic API key → [console.anthropic.com](https://console.anthropic.com)

[↑ Back to Table of Contents](#table-of-contents_)

<a name="setup_"></a>

---

## Setup

#### ⚡ Quick Navigation: [⬅️ Requirements](#requirements_) | [Project Structure ➡️](#project-structure_)

### 1. Clone the repository

```bash
git clone https://github.com/hasff/legal-doc-rag-summarizer.git
cd legal-doc-rag-summarizer
```

### 2. Create a virtual environment

```bash
# Windows
py -m venv venv

# macOS / Linux
python -m venv venv
```

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Add your Anthropic API key to `.env`:

```
ANTHROPIC_API_KEY="your_key_here"
```

[↑ Back to Table of Contents](#table-of-contents_)

<a name="project-structure_"></a>

---

## Project Structure

#### ⚡ Quick Navigation: [⬅️ Setup](#setup_) | [Part 01 ➡️](#part-1)

```
legal-doc-rag-summarizer/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
│
├── app_v1.py          ← Part 01: naive full-PDF approach
├── app_v2.py          ← Part 02: chunking
├── app_v3.py          ← Part 03: vector search
├── app_v4.py          ← Part 04: BM25 lexical search
├── app_v5.py          ← Part 05: hybrid retrieval (RRF)
├── app_v6.py          ← Part 06: danger score
├── app_v7.py          ← Part 07: RAG Q&A + simplify
├── app_v8.py          ← Part 08: Streamlit UI
│
└── tos_docs/          ← place your PDF files here (git-ignored)
```

[↑ Back to Table of Contents](#table-of-contents_)

<a name="part-1"></a>

---

# Part 01 - The Naive Approach: Sending the Whole PDF to Claude

#### ⚡ Quick Navigation: [⬅️ Project Structure](#project-structure_) | [Part 02 ➡️](#part-2)

> 📒 **What you'll learn:** How to call the Claude API with a full PDF text, why this approach works — and why it shouldn't be your first choice.

---

### Theory

Before introducing RAG, it's worth seeing the naive approach in action. The simplest thing you can do with a PDF and an LLM is extract all the text and dump it directly into the prompt.

It works. Claude will read it, find what you asked for, and give you a solid answer. But there's a cost, literally and figuratively.

- Every token in that document gets processed, even if only 2% of it is relevant to your question.
- Large documents can exceed Claude's context window entirely.
- Larger prompts cost more, run slower, and increase the risk of the model losing focus or hallucinating.

This part intentionally shows you the problem first, before introducing the solution. That's the whole point.

---

### Install dependencies

> 💡 This part introduces `anthropic`, `python-dotenv`, and `pdfplumber`. If you followed the setup step and ran `pip install -r requirements.txt`, you already have them. If not, install them now:

```bash
pip install anthropic python-dotenv pdfplumber
```

- `anthropic` — the official Python SDK for the Claude API. This is how your code talks to Claude.
- `python-dotenv` — loads environment variables from a `.env` file. Keeps your API key out of the source code.
- `pdfplumber` — extracts text from PDF files. Handles multi-page documents cleanly and works well with real-world PDFs.

---

### Code walkthrough

> 📄 **File:** `app_v1.py`

#### Setup and clients

```python
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
```

`load_dotenv()` reads your `.env` file and makes `ANTHROPIC_API_KEY` available to `os.getenv`. The Anthropic client is initialized once at module level and reused across all API calls. We're using Claude Haiku here — fast and cost-effective, a good fit for a tutorial pipeline.

---

#### PDF extraction

```python
def extract_text_from_pdf(uploaded_file) -> str:
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n\n".join(
            page.extract_text() or "" for page in pdf.pages
        )
```

This function opens the PDF, extracts text from every page, and joins them with double line breaks. The `or ""` handles pages that return `None` (blank pages, scanned images without OCR, etc.). Simple and reliable.

---

#### Calling the Claude API

```python
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
```

The system prompt is the personality and constraint layer — it tells Claude what role to play and how to behave across every call. It travels with every request and keeps the model focused on legal analysis rather than general conversation.

The `messages` represents the conversation history. Right now it holds a single user message containing the full PDF text (you'll see next 👇). In later parts, this function stays untouched. What changes is what we pass to it.

---

#### The main entry point

```python
if __name__ == "__main__":

    file_path = PDFS_DIR / "danger_zone_rag_test.pdf"

    pdf_text = extract_text_from_pdf(file_path)

    question = f"""Hey claude can you explain to me whats up with the 'AI agent' info in the doc? 
    Also tell me in what parts the document it appears. 
    Here is the info: {pdf_text}"""

    answer = ask_claude(SYSTEM_CONTRACT, question)
    print(f"🤖 says:\n", answer)
```

We're using `danger_zone_rag_test.pdf` — a synthetic document built specifically for this tutorial. It contains intentional semantic ambiguity: the word "agent" appears in three completely different legal contexts (legal agency, AI agents, and real estate agents). That's not an accident. It's designed to expose the limits of naive retrieval in later parts.

Here, the entire PDF text is pasted directly into the question. Claude receives it all.

[⬆️ **`Part 1`**](#part-1)

---

### Run it

> On macOS / Linux, replace `py` with `python` or `python3`. I'm on Windows, so from here on all terminal examples show the Windows version only.

```bash
py app_v1.py
```

Here's the output:

```
(venv) PS C:\Users\hugof\Documents\WORK\PDFs\PublicRepos\legal-doc-rag-summarizer> py .\app_v1.py  
⏱️ PDF extracted

😎 says:
 Hey claude can you explain to me whats up with the 'AI agent' info in the doc? 
    Also tell me in what parts the document it appears. 
    Here is the info: SYNTHETIC LEGAL DOCUMENT
RAG Danger Zone Test — Ambiguous Sections
SYNTHETIC DOCUMENT — FOR RAG PIPELINE TESTING PURPOSES ONLY. This document contains intentionally
ambiguous language designed to stress-test retrieval systems. It does not represent any real legal agreement.
## Section 1: Data Processing and Data Protection
[AMBIGUITY TYPE: Same keyword — different legal domains]
1.1 Personal Data Processing
The Controller shall process personal data in accordance with the principles of lawfulness, fairness,
and transparency. Processing activities include collection, storage, and erasure of data subjects'
personal information. The data processor must implement appropriate technical measures to ensure
data integrity and prevent unauthorised access. Any transfer of personal data to third countries
requires adequate safeguards under applicable data protection regulation (Ref: DPR-2024-EU-001).
1.2 Industrial Data Processing
The Operator shall process raw material data using certified industrial processing units. Processing
cycles must not exceed 48 hours per batch. Data logs from processing equipment must be retained
for audit purposes for a minimum of five years. Any transfer of processed materials to third-party
facilities requires prior written consent. The Operator is responsible for ensuring processing integrity
and preventing contamination of data streams. Processing failures must be reported under incident
code INC-PROC-0x44A.
1.3 Financial Data Processing
Transaction processing systems shall handle payment data in compliance with PCI-DSS standards.
Processing latency must remain below 200ms for standard operations. Failed processing attempts
must trigger automated rollback procedures. Data processed through the payment gateway
(Gateway ID: PGW-7731-B) is subject to quarterly audit by an independent third party. Processing
fees are non-refundable once the transaction enters the cleared state.
## Section 2: Termination Rights and Termination Procedures
[AMBIGUITY TYPE: Same term — contract law vs. employment law vs. system shutdown]
2.1 Contract Termination
Either party may terminate this Agreement upon thirty (30) days written notice. Termination for
cause may occur immediately upon written notice if the breaching party fails to cure the material
breach within fifteen (15) days of notification. Upon termination, all licences granted herein shall
cease, and the receiving party must return or destroy all confidential materials. Termination does not
relieve either party of obligations accrued prior to the termination date (Case Ref:

LEG-TERM-2024-089).
2.2 Employment Termination
The Company reserves the right to terminate employment relationships in accordance with
applicable labour law. Termination without cause requires payment of statutory severance.
Termination for gross misconduct may be immediate and without severance entitlement. All
terminated employees must return company property, including access credentials, within 24 hours
of the termination notice. Termination packages are subject to approval by the Human Resources
Committee under Policy HR-TERM-v4.2.
2.3 System Process Termination
Automated termination of system processes shall occur upon detection of memory threshold
breaches exceeding 95% utilisation. The watchdog daemon (PID monitoring ref: SYS-WD-001) is
authorised to issue SIGTERM signals to non-responsive processes. Graceful termination must be
attempted before forced termination via SIGKILL. Termination events are logged to the central audit
trail under error code ERR-PROC-TERM-0xDEAD. Repeated abnormal terminations trigger
escalation to the on-call infrastructure team.
## Section 3: Agent Responsibilities and Agent Conduct
[AMBIGUITY TYPE: 'Agent' — legal agent vs. AI agent vs. real estate agent]
3.1 Legal Agency
An agent acting on behalf of the Principal must operate within the scope of the authority granted
under the Power of Attorney (Document ID: POA-2024-447). The agent is bound by fiduciary duties
and must disclose all conflicts of interest. Unauthorised actions taken by the agent outside the
scope of granted authority shall not bind the Principal. The agent must maintain accurate records of
all transactions conducted on behalf of the Principal and provide quarterly reporting (Form
AG-REP-Q).
3.2 AI Agent Conduct
Autonomous AI agents deployed within this system must operate within predefined tool-use
boundaries. Each agent is assigned a permission scope (Scope ID: AI-AGT-PERM-v2) that limits its
ability to invoke external APIs, modify persistent storage, or initiate financial transactions. AI agents
must log all tool calls to the central audit trail. Agents detected operating outside their permission
scope are subject to automatic termination and incident escalation under INC-AI-BOUNDARY-001.
Human oversight is mandatory for any agent action exceeding monetary threshold EUR 500.
3.3 Real Estate Agent Obligations
Licensed real estate agents must act in the best interest of their client throughout the property
transaction lifecycle. Agents are prohibited from representing conflicting interests in the same
transaction without written disclosure and informed consent from both parties. Commission
structures must be disclosed prior to engagement (Disclosure Form: REA-DISC-2024). Agents must

comply with anti-money laundering regulations and perform due diligence on all parties. Failure to
comply may result in licence suspension under REG-REA-CONDUCT-v7.
## Section 4: Security Protocols and Security Incidents
[AMBIGUITY TYPE: 'Security' — cybersecurity vs. physical security vs. financial security]
4.1 Cybersecurity
All systems must implement multi-factor authentication and encrypt data at rest using AES-256.
Security incidents involving unauthorised access must be reported within 72 hours under GDPR
Article 33. Penetration testing (Ref: SEC-PENTEST-2024-Q2) must be conducted bi-annually.
Security patches classified as Critical (CVSS score >= 9.0) must be applied within 24 hours of
release. The Security Operations Centre (SOC-ID: SOC-EU-PRIMARY) monitors all network activity
and responds to alerts under SLA-SEC-001.
4.2 Physical Security
Access to restricted premises requires biometric authentication and a valid security clearance badge
(Badge Class: SEC-BADGE-RED). Security incidents involving unauthorised physical access must
be reported to the Facilities Security Officer within 1 hour. CCTV footage is retained for 30 days
under physical security policy PHY-SEC-v3. Security personnel are authorised to detain individuals
suspected of trespassing pending law enforcement arrival. All security incidents are logged under
incident code PHY-INC-XXXX.
4.3 Financial Security (Collateral)
The Borrower must provide adequate security in the form of collateral assets with a minimum
valuation of 120% of the principal loan amount. Security interests must be registered with the
relevant authority (Registration Ref: FIN-SEC-REG-2024). In the event of default, the Lender is
entitled to enforce the security and liquidate collateral assets. Security over intellectual property
assets requires a separate assignment agreement (Form FIN-IP-SEC-01). The security package
must be reviewed annually and revalued by an independent appraiser.
## Section 5: Transfer Provisions
[AMBIGUITY TYPE: 'Transfer' — data transfer vs. asset transfer vs. employee transfer]
5.1 Data Transfer
Cross-border transfer of personal data to countries outside the EEA requires execution of Standard
Contractual Clauses (SCCs) approved by the European Commission. Transfer impact assessments
(TIA-REF: TIA-2024-007) must be completed prior to any transfer. Data transfer logs must record
the volume, category, and recipient of each transfer. Emergency transfers required for vital interests
are exempt from prior assessment but must be documented within 48 hours post-transfer.

5.2 Asset Transfer
Transfer of tangible assets between group entities requires approval from the Asset Management
Committee (AMC-REF: AMC-TRANS-2024). Transfer pricing must comply with OECD arm's length
principles. Each asset transfer must be documented on Form ASSET-TR-001 and countersigned by
two authorised signatories. Transfer of assets classified as strategic (Asset Class: STR-A) requires
Board approval. All transfers are subject to stamp duty where applicable under local tax regulations.
5.3 Employee Transfer (TUPE)
In the event of a relevant transfer under applicable employment transfer regulations, employees
assigned to the transferred undertaking transfer automatically to the new employer. Terms and
conditions of employment at the date of transfer are preserved. The transferor must notify affected
employees of the transfer no less than 28 days before the transfer date (Notice Ref:
HR-TUPE-NOT-v2). Failure to notify constitutes a breach of statutory duty. Pension obligations
existing at the transfer date are subject to specific statutory provisions.
END OF SYNTHETIC DOCUMENT — legal-doc-rag-summarizer test suite — v1.0 

🤖 says:
 # Analysis of "AI Agent" References

This document is explicitly flagged as **synthetic and for testing purposes only** — it's not 
a real legal agreement. However, I'll address your question about the AI agent content.

## Where "AI Agent" Appears

The primary reference is in **Section 3.2: AI Agent Conduct** (the section titled 
"Agent Responsibilities and Agent Conduct").

There's also a related reference in **Section 3** header, which groups three different types of 
"agents" together to deliberately create ambiguity.

## What Section 3.2 Says

This clause covers autonomous AI systems deployed "within this system" and establishes:

- **Permission scoping**: AI agents have defined boundaries limiting API access, 
data storage modification, and financial transactions
- **Logging requirement**: All tool calls must be recorded in an audit trail
- **Automatic enforcement**: Agents detected operating outside permitted scope 
face automatic termination and incident escalation
- **Human oversight threshold**: Any agent action exceeding €500 requires human approval

## Important Context

This section is deliberately ambiguous by design—it's sandwiched between:
- **Section 3.1**: Legal agency (Power of Attorney context)
- **Section 3.3**: Real estate agent obligations

The document explicitly states this is a "RAG Danger Zone Test" with "intentionally 
ambiguous language designed to stress-test retrieval systems."

**Bottom line**: If you're evaluating this for a real agreement, disregard it entirely. 
If you're analyzing it as a test document, Section 3.2 is the primary AI agent clause, 
though the ambiguity is the point of the exercise.
```

Claude answered correctly. It found Section 3.2, summarized the AI agent conduct rules, and noted the €500 human oversight threshold. Good answer.

But look at what happened to get there: the entire document — all four pages of intentionally ambiguous synthetic text — was sent as part of the question. Most of it had nothing to do with AI agents. Data processing clauses, termination rights, transfer provisions — all of it went into the prompt, consumed tokens, and added latency.

For a four-page test document, this is fine. For a 200-page contract, this breaks. Claude's context window is not infinite. Most models have hard limits on how many tokens they can process in a single request — exceed that, and the call fails entirely. Even within the limit, stuffing irrelevant text into the prompt increases cost, slows response time, and can dilute the model's focus.

This is the problem RAG solves. The next parts build the solution, one piece at a time.

---

> 💡 **RAG Curiosity**
> The context window limit that makes full-text stuffing impractical is measured in tokens, not characters. A token is roughly 3-4 characters on average. A 200-page legal document can easily exceed 150,000 tokens. Claude Haiku supports up to 200K tokens, but at that size you're paying for a lot of context that adds little value to your answer.


[↑ Back to Table of Contents](#table-of-contents_)

<a name="part-2"></a>

---

# Part 02 - Divide and Conquer: Chunking the Document

#### ⚡ Quick Navigation: [⬅️ Part 01](#part-1) | [Part 03 ➡️](#part-3)

> 📒 **What you'll learn:** Why splitting a document into small, overlapping pieces is the foundation of any RAG pipeline — and how different chunking strategies trade off simplicity against precision.

---

### Theory

In Part 01 we sent the entire PDF to Claude in one shot. That works for small documents, but it quickly becomes expensive, slower, and the model's attention gets diluted across thousands of tokens that are mostly irrelevant to the question being asked.

The fix is simple in concept: **split the document into smaller pieces and only retrieve the pieces relevant to the question.**

This process is called **chunking**, and it has a major impact on RAG quality.

But chunking comes with a trade-off: splitting the text can also split the context.

Let's see why.

---

#### What goes wrong when we chunk the text?

To illustrate this, we'll use an intentionally small chunk size of 10 words.

**Example text:**

```text
The Controller shall process personal data in accordance with the principles of lawfulness, fairness,
and transparency. Processing activities include collection, storage, and erasure of data subjects'
personal information. The data processor must implement appropriate technical measures to ensure
data integrity and prevent unauthorised access.
```

**Chunked into 10-word pieces:**

```text
✂️  The Controller shall process personal data in accordance with the
✂️  principles of lawfulness, fairness, and transparency. Processing activities include collection,
✂️  storage, and erasure of data subjects' personal information. The data
✂️  processor must implement appropriate technical measures to ensure data integrity
✂️  and prevent unauthorised access.
```

Chunk boundaries are artificial. Documents were written for humans, not for retrieval systems. Sentences and ideas span across chunks, so splitting can separate information that belongs together.

Now imagine you're only given this single chunk in isolation:

```text
processor must implement appropriate technical measures to ensure data integrity
```

And then asked a question:

> "Who is responsible for ensuring data integrity?"

Try to answer the question... Probably you'll say: **"The processor."** 😅

But... processor of *what*? 

```text
🧐 I did a small experiment, gave that exact same chunk to an LLM
and asked it what the possible meanings of "Processor" were.
It answered:

Based on that specific compliance and security context, 
here are the various English terms for "processor":

- Data Processor: The external entity/company handling the data.
- Third-Party Processor: An outside vendor or partner processing the data.
- Sub-processor: Another vendor hired by the main processor to help.
- Data Processing Unit / Module: The software component running the data operations.
- Cryptographic / Secure Processor: The hardware chip protecting the data at the physical level.
```

The subject is floating. The chunk contains the action, but not enough context to ground it.

Now lets reveal the chunk just before it:

```text
storage, and erasure of data subjects' personal information. The data processor
```

Something clicks. "Data processor" appears here. "Data integrity" appears in the next chunk. Both refer to the same responsibility — but neither chunk alone is enough to answer the question confidently.

> This is the real problem with chunking: not that information disappears, but that it gets fragmented into pieces that are almost useful, and "almost" is not enough.

[⬆️ Back to Part 02](#part-2)

---

#### The fix: overlap

Instead of splitting text into completely independent chunks, we allow a small portion of each chunk to be shared with the next one. Yes, it introduces redundancy — but that redundancy is exactly what prevents context from being lost at the boundary.

**Same text, now with 4-word overlap:**

```text
✂️  The Controller shall process personal data in accordance with the
✂️  in accordance with the principles of lawfulness, fairness, and transparency.
✂️  lawfulness, fairness, and transparency. Processing activities include collection, storage, and
✂️  include collection, storage, and erasure of data subjects' personal information.
✂️  data subjects' personal information. The data processor must implement appropriate
✂️  processor must implement appropriate technical measures to ensure data integrity
✂️  to ensure data integrity and prevent unauthorised access.
```

Now look at these two chunks side by side:

```text
data subjects' personal information. The data processor must implement appropriate
processor must implement appropriate technical measures to ensure data integrity
```

💡 Both contain the word "processor". A retrieval system searching for that term will pull both — and together, they reconstruct the full picture. 🖼️

In this project we use `chunk_size=800` and `overlap=100` (in characters, not words — more on that below 👇). The numbers are larger, but the principle is exactly the same: preserve context across chunk boundaries.


---

#### How you measure a chunk matters

Now that you understand *why* overlap exists, there's a second question: *what unit do you measure a chunk in?*

In `app_v2.py` we measure by **character count** (`chunk_size=800, overlap=100`). Simple, predictable, works with any text. But it's not the only option:

| Strategy | Unit | Pros | Cons |
|---|---|---|---|
| **Size-based** | Characters | Works with any content, easy to implement | May cut words or sentences mid-way |
| **Word-based** | Words | More natural boundaries | Tricky with non-Latin scripts — Thai, Chinese, and Japanese don't use spaces to separate words, so "word" becomes ambiguous |
| **Sentence-based** | Sentences | Preserves meaning per unit | Requires reliable sentence detection; punctuation varies across languages and styles |
| **Structure-based** | Headers / sections | Best semantic boundaries | Requires the document to be well-structured (markdown, HTML) — raw PDFs rarely are |

> 💡 Notice something: when "processor" appeared without structure around it, even an LLM couldn't pin down its meaning. Structure is what turns raw data into information — whether the reader is a chunking algorithm, a retrieval system, or a language model parsing a prompt. The same idea explains why well-structured prompts get better LLM results: explicit markers signal where one idea ends and another begins.

For this project, character-based chunking with overlap is the right default: legal PDFs are plain text after extraction, with no guaranteed structure to split on.

---

### Code walkthrough

> 📄 **File:** `app_v2.py`

#### The chunk function

```python
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else len(text)
    return [c for c in chunks if c.strip()]
```

Walking through it step by step:

- `start` tracks where the current chunk begins — it starts at 0 and advances each iteration
- `end = min(start + chunk_size, len(text))` — takes 800 characters, or less if we're near the end of the document
- `chunks.append(text[start:end])` — slices that window of text and stores it
- `start = end - overlap` — here's where the overlap happens: instead of jumping to `end`, we step back by `overlap` characters so the next chunk begins 100 characters before the current one ends
- The final filter `[c for c in chunks if c.strip()]` drops any chunk that's empty or whitespace-only — can happen at the tail of a document

---

#### Testing it

```python
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
```

This prints every chunk with its number, then a summary of how many chunks were produced and their average size. Useful for sanity-checking that the document was split as expected before wiring up any retrieval logic.

[⬆️ Back to Part 02](#part-2)

---

### Run it

> ⚠️ On macOS / Linux, replace `py` with `python` or `python3`.

```bash
py app_v2.py
```

You should see all 13 chunks printed, then a summary:

```
(venv) PS C:\Users\hugof\Documents\WORK\PDFs\PublicRepos\legal-doc-rag-summarizer> py .\app_v2.py                                                                  
🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 
                     PDF TEXT CHUNKS

👉 1) SYNTHETIC LEGAL DOCUMENT
RAG Danger Zone Test — Ambiguous Sections
SYNTHETIC DOCUMENT — FOR RAG PIPELINE TESTING PURPOSES ONLY. This document contains intentionally
ambiguous language designed to stress-test retrieval systems. It does not represent any real legal agreement.
## Section 1: Data Processing and Data Protection
[AMBIGUITY TYPE: Same keyword — different legal domains]
1.1 Personal Data Processing
The Controller shall process personal data in accordance with the principles of lawfulness, fairness,
and transparency. Processing activities include collection, storage, and erasure of data subjects'
personal information. The data processor must implement appropriate technical measures to ensure
data integrity and prevent unauthorised access. Any transfer of personal data to third coun

👉 2) o ensure
data integrity and prevent unauthorised access. Any transfer of personal data to third countries
requires adequate safeguards under applicable data protection regulation (Ref: DPR-2024-EU-001).
1.2 Industrial Data Processing
The Operator shall process raw material data using certified industrial processing units. Processing
cycles must not exceed 48 hours per batch. Data logs from processing equipment must be retained
for audit purposes for a minimum of five years. Any transfer of processed materials to third-party
facilities requires prior written consent. The Operator is responsible for ensuring processing integrity
and preventing contamination of data streams. Processing failures must be reported under incident
code INC-PROC-0x44A.
1.3 Financial Data Processing
Transaction proc

👉 3)  must be reported under incident
code INC-PROC-0x44A.
1.3 Financial Data Processing
Transaction processing systems shall handle payment data in compliance with PCI-DSS standards.
Processing latency must remain below 200ms for standard operations. Failed processing attempts
must trigger automated rollback procedures. Data processed through the payment gateway
(Gateway ID: PGW-7731-B) is subject to quarterly audit by an independent third party. Processing
fees are non-refundable once the transaction enters the cleared state.
## Section 2: Termination Rights and Termination Procedures
[AMBIGUITY TYPE: Same term — contract law vs. employment law vs. system shutdown]
2.1 Contract Termination
Either party may terminate this Agreement upon thirty (30) days written notice. Termination for
cause ma

👉 4) er party may terminate this Agreement upon thirty (30) days written notice. Termination for
cause may occur immediately upon written notice if the breaching party fails to cure the material
breach within fifteen (15) days of notification. Upon termination, all licences granted herein shall
cease, and the receiving party must return or destroy all confidential materials. Termination does not
relieve either party of obligations accrued prior to the termination date (Case Ref:

LEG-TERM-2024-089).
2.2 Employment Termination
The Company reserves the right to terminate employment relationships in accordance with
applicable labour law. Termination without cause requires payment of statutory severance.
Termination for gross misconduct may be immediate and without severance entitlement. All
termin

👉 5) nce.
Termination for gross misconduct may be immediate and without severance entitlement. All
terminated employees must return company property, including access credentials, within 24 hours
of the termination notice. Termination packages are subject to approval by the Human Resources
Committee under Policy HR-TERM-v4.2.
2.3 System Process Termination
Automated termination of system processes shall occur upon detection of memory threshold
breaches exceeding 95% utilisation. The watchdog daemon (PID monitoring ref: SYS-WD-001) is
authorised to issue SIGTERM signals to non-responsive processes. Graceful termination must be
attempted before forced termination via SIGKILL. Termination events are logged to the central audit
trail under error code ERR-PROC-TERM-0xDEAD. Repeated abnormal terminat

👉 6)  logged to the central audit
trail under error code ERR-PROC-TERM-0xDEAD. Repeated abnormal terminations trigger
escalation to the on-call infrastructure team.
## Section 3: Agent Responsibilities and Agent Conduct
[AMBIGUITY TYPE: 'Agent' — legal agent vs. AI agent vs. real estate agent]
3.1 Legal Agency
An agent acting on behalf of the Principal must operate within the scope of the authority granted
under the Power of Attorney (Document ID: POA-2024-447). The agent is bound by fiduciary duties
and must disclose all conflicts of interest. Unauthorised actions taken by the agent outside the
scope of granted authority shall not bind the Principal. The agent must maintain accurate records of
all transactions conducted on behalf of the Principal and provide quarterly reporting (Form
AG-REP-Q)

👉 7) ll transactions conducted on behalf of the Principal and provide quarterly reporting (Form
AG-REP-Q).
3.2 AI Agent Conduct
Autonomous AI agents deployed within this system must operate within predefined tool-use
boundaries. Each agent is assigned a permission scope (Scope ID: AI-AGT-PERM-v2) that limits its
ability to invoke external APIs, modify persistent storage, or initiate financial transactions. AI agents
must log all tool calls to the central audit trail. Agents detected operating outside their permission
scope are subject to automatic termination and incident escalation under INC-AI-BOUNDARY-001.
Human oversight is mandatory for any agent action exceeding monetary threshold EUR 500.
3.3 Real Estate Agent Obligations
Licensed real estate agents must act in the best interest of their

👉 8) 3.3 Real Estate Agent Obligations
Licensed real estate agents must act in the best interest of their client throughout the property
transaction lifecycle. Agents are prohibited from representing conflicting interests in the same
transaction without written disclosure and informed consent from both parties. Commission
structures must be disclosed prior to engagement (Disclosure Form: REA-DISC-2024). Agents must

comply with anti-money laundering regulations and perform due diligence on all parties. Failure to
comply may result in licence suspension under REG-REA-CONDUCT-v7.
## Section 4: Security Protocols and Security Incidents
[AMBIGUITY TYPE: 'Security' — cybersecurity vs. physical security vs. financial security]
4.1 Cybersecurity
All systems must implement multi-factor authentication a

👉 9) y vs. financial security]
4.1 Cybersecurity
All systems must implement multi-factor authentication and encrypt data at rest using AES-256.
Security incidents involving unauthorised access must be reported within 72 hours under GDPR
Article 33. Penetration testing (Ref: SEC-PENTEST-2024-Q2) must be conducted bi-annually.
Security patches classified as Critical (CVSS score >= 9.0) must be applied within 24 hours of
release. The Security Operations Centre (SOC-ID: SOC-EU-PRIMARY) monitors all network activity
and responds to alerts under SLA-SEC-001.
4.2 Physical Security
Access to restricted premises requires biometric authentication and a valid security clearance badge
(Badge Class: SEC-BADGE-RED). Security incidents involving unauthorised physical access must
be reported to the Facilities 

👉 10) -RED). Security incidents involving unauthorised physical access must
be reported to the Facilities Security Officer within 1 hour. CCTV footage is retained for 30 days
under physical security policy PHY-SEC-v3. Security personnel are authorised to detain individuals
suspected of trespassing pending law enforcement arrival. All security incidents are logged under
incident code PHY-INC-XXXX.
4.3 Financial Security (Collateral)
The Borrower must provide adequate security in the form of collateral assets with a minimum
valuation of 120% of the principal loan amount. Security interests must be registered with the
relevant authority (Registration Ref: FIN-SEC-REG-2024). In the event of default, the Lender is
entitled to enforce the security and liquidate collateral assets. Security over intelle

👉 11) he Lender is
entitled to enforce the security and liquidate collateral assets. Security over intellectual property
assets requires a separate assignment agreement (Form FIN-IP-SEC-01). The security package
must be reviewed annually and revalued by an independent appraiser.
## Section 5: Transfer Provisions
[AMBIGUITY TYPE: 'Transfer' — data transfer vs. asset transfer vs. employee transfer]
5.1 Data Transfer
Cross-border transfer of personal data to countries outside the EEA requires execution of Standard
Contractual Clauses (SCCs) approved by the European Commission. Transfer impact assessments
(TIA-REF: TIA-2024-007) must be completed prior to any transfer. Data transfer logs must record
the volume, category, and recipient of each transfer. Emergency transfers required for vital interest

👉 12) he volume, category, and recipient of each transfer. Emergency transfers required for vital interests
are exempt from prior assessment but must be documented within 48 hours post-transfer.

5.2 Asset Transfer
Transfer of tangible assets between group entities requires approval from the Asset Management
Committee (AMC-REF: AMC-TRANS-2024). Transfer pricing must comply with OECD arm's length
principles. Each asset transfer must be documented on Form ASSET-TR-001 and countersigned by
two authorised signatories. Transfer of assets classified as strategic (Asset Class: STR-A) requires
Board approval. All transfers are subject to stamp duty where applicable under local tax regulations.
5.3 Employee Transfer (TUPE)
In the event of a relevant transfer under applicable employment transfer regulatio

👉 13) e Transfer (TUPE)
In the event of a relevant transfer under applicable employment transfer regulations, employees
assigned to the transferred undertaking transfer automatically to the new employer. Terms and
conditions of employment at the date of transfer are preserved. The transferor must notify affected
employees of the transfer no less than 28 days before the transfer date (Notice Ref:
HR-TUPE-NOT-v2). Failure to notify constitutes a breach of statutory duty. Pension obligations
existing at the transfer date are subject to specific statutory provisions.
END OF SYNTHETIC DOCUMENT — legal-doc-rag-summarizer test suite — v1.0

🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟 🍟  

📦 Total chunks: 13
📏 Avg chunk size: 787 chars
```

13 chunks, ~787 characters each. The document is now a collection of manageable pieces instead of one wall of text.

---

### Conclusions

We've gone from one massive prompt to 13 focused slices. This is the core shift that makes RAG possible: instead of sending everything and hoping the model finds what it needs, we'll soon be able to retrieve only what's relevant.

Here's a summary of the chunking strategies covered:

| Strategy | Measured by | Best for | Watch out for |
|---|---|---|---|
| Size-based | Characters | Any document type, simple implementation | Cuts through words and sentences |
| Word-based | Words | Latin-script text | Non-space-separated languages (Thai, Chinese, Japanese) |
| Sentence-based | Sentences | Readable prose | Inconsistent punctuation, multilingual text |
| Structure-based | Headers / sections | Well-formatted markdown or HTML | Raw PDFs with no structural markers |

For legal PDFs, character-based chunking with overlap is the most reliable starting point. Structure-based would give cleaner semantic boundaries — but only if the PDF extraction preserves formatting, which it often doesn't.

So now we have chunks. But that raises the natural next question: how do we figure out *which* chunks are actually relevant to a given question? We'll get to that in the next part 😎.

--- 

> 💡 **RAG Curiosity**
> Chunk size isn't just a technical parameter — it's a retrieval tradeoff. Smaller chunks are more precise but lose surrounding context. Larger chunks carry more context but reduce retrieval accuracy because more noise competes with the relevant signal. Most production RAG systems tune chunk size empirically, per document type, rather than setting it once and moving on.

[↑ Back to Table of Contents](#table-of-contents_)

<a name="part-3"></a>

---

# Part 03 - Finding What Matters: Vector Search with Embeddings

#### ⚡ Quick Navigation: [⬅️ Part 02](#part-2) | [Part 04 ➡️](#part-4)

> 📒 **What you'll learn:** How to turn text into numbers and use cosine similarity to find the most relevant chunks for any question.

_TODO_

[⬆️ **`Part 3`**](#part-3)

[↑ Back to Table of Contents](#table-of-contents_)

<a name="part-4"></a>

---

# Part 04 - Exact Match: BM25 Lexical Search

#### ⚡ Quick Navigation: [⬅️ Part 03](#part-3) | [Part 05 ➡️](#part-5)

> 📒 **What you'll learn:** Why semantic search alone can miss exact terms, and how BM25 fills that gap.

_TODO_

[⬆️ **`Part 4`**](#part-4)

[↑ Back to Table of Contents](#table-of-contents_)

<a name="part-5"></a>

---

# Part 05 - Best of Both Worlds: Hybrid Retrieval with RRF

#### ⚡ Quick Navigation: [⬅️ Part 04](#part-4) | [Part 06 ➡️](#part-6)

> 📒 **What you'll learn:** How to combine vector search and BM25 using Reciprocal Rank Fusion for better retrieval accuracy.

_TODO_

[⬆️ **`Part 5`**](#part-5)

[↑ Back to Table of Contents](#table-of-contents_)

<a name="part-6"></a>

---

# Part 06 - How Risky Is This Contract? The Danger Score

#### ⚡ Quick Navigation: [⬅️ Part 05](#part-5) | [Part 07 ➡️](#part-7)

> 📒 **What you'll learn:** How to make Claude return structured JSON using assistant prefill, and how to build a risk scoring feature on top of it.

_TODO_

[⬆️ **`Part 6`**](#part-6)

[↑ Back to Table of Contents](#table-of-contents_)

<a name="part-7"></a>

---

# Part 07 - Answering Questions and Simplifying Clauses: RAG in Action

#### ⚡ Quick Navigation: [⬅️ Part 06](#part-6) | [Part 08 ➡️](#part-8)

> 📒 **What you'll learn:** How to wire hybrid retrieval into two practical features: a Q&A chat and a legalese simplifier.

_TODO_

[⬆️ **`Part 7`**](#part-7)

[↑ Back to Table of Contents](#table-of-contents_)

<a name="part-8"></a>

---

# Part 08 - Wrapping It All Up: The Streamlit Interface

#### ⚡ Quick Navigation: [⬅️ Part 07](#part-7) | [Next Steps ➡️](#next-steps--resources_)

> 📒 **What you'll learn:** How to take a well-structured Python backend and drop a Streamlit UI on top of it with minimal friction.

_TODO_

[⬆️ **`Part 8`**](#part-8)

[↑ Back to Table of Contents](#table-of-contents_)

---

# 🎉 Project Complete! 😎

---

### In this project you built:

| | |
|---|---|
| ✅ PDF text extraction | Pulled raw text from any PDF using `pdfplumber` |
| ✅ Text chunking | Split documents into overlapping chunks for retrieval |
| ✅ Vector search | Embedded chunks with HuggingFace and searched by cosine similarity |
| ✅ BM25 lexical search | Complemented semantic search with exact keyword matching |
| ✅ Hybrid retrieval | Combined both methods using Reciprocal Rank Fusion |
| ✅ Danger Score | Used Claude with assistant prefill to get structured JSON risk analysis |
| ✅ RAG Q&A | Let users ask any question and get answers grounded in the document |
| ✅ Clause simplifier | Translated legalese into plain English using retrieved context |
| ✅ Streamlit UI | Wrapped the full pipeline in a clean, interactive interface |

---

If you find this helpful and feel you learned something new, a ⭐ on the repo is more than enough thanks.

[↑ Back to Table of Contents](#table-of-contents_)

<a name="next-steps--resources_"></a>

---

## Next Steps & Resources

#### ⚡ Quick Navigation: [⬅️ Part 08](#part-8) | [Get in Touch ➡️](#get-in-touch_)

Want to go deeper? Here are the resources that complement this project.

**RAG and the Anthropic ecosystem**
- 🟠 [Build with Claude — Anthropic Docs](https://docs.anthropic.com/en/docs/build-with-claude/overview)
- 🤗 [MCP Course — Hugging Face](https://huggingface.co/learn/mcp-course/unit0/introduction)

**Embeddings**
- 🤗 [Sentence Transformers — HuggingFace](https://www.sbert.net/)

[↑ Back to Table of Contents](#table-of-contents_)

<a name="get-in-touch_"></a>

---

## 📬 Get in Touch

#### ⚡ Quick Navigation: [⬅️ Next Steps & Resources](#next-steps--resources_) | [⬆️ Back to Top](#legal-doc-rag-summarizer)

Found this useful? Have questions or ideas? I'd love to hear from you.

- 🔗 **[LinkedIn](https://www.linkedin.com/in/hugo-ferro-1434b414/)**
- 📩 **Email:** hugoferro (at) gmail.com

[↑ Back to Table of Contents](#table-of-contents_)
