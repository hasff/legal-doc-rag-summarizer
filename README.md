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

> 📒 **What you'll learn:** How to split a document into manageable chunks so only the relevant pieces go into the prompt.

_TODO_

[⬆️ **`Part 2`**](#part-2)

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
