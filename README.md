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
├── app_full.py        ← final consolidated version
│
└── tos_docs/          ← place your PDF files here (git-ignored)
```

[↑ Back to Table of Contents](#table-of-contents_)

<a name="part-1"></a>

---

# Part 01 - The Naive Approach: Sending the Whole PDF to Claude

#### ⚡ Quick Navigation: [⬅️ Project Structure](#project-structure_) | [Part 02 ➡️](#part-2)

> 📒 **What you'll learn:** Why sending an entire PDF to Claude works, and why it doesn't scale.

_TODO_

[⬆️ **`Part 1`**](#part-1)

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
