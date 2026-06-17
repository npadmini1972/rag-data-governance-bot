# FDA Part 11 RAG Governance Q&A Bot
**A Data Governance tool powered by Claude AI + RAG**

> Live demo: [(https://pnagarajan-rag-data-governance-bot.streamlit.app/)]((https://pnagarajan-rag-data-governance-bot.streamlit.app/))

## Problem Statement
Data governance and compliance teams at pharma and healthcare companies 
maintain large libraries of regulatory policies. Analysts waste time 
searching through dense PDFs to answer questions like "What are the audit 
trail requirements?" or "How does FDA define electronic signatures?" 
This RAG bot answers those questions in plain English, with citations 
back to the source document — in under 10 seconds.

## What It Does
- Accepts plain-English questions about FDA 21 CFR Part 11
- Retrieves the most relevant sections from the guidance document
- Generates grounded answers using Claude AI
- Cites the exact source page for every answer
- Refuses to answer questions outside the document scope (hallucination guardrail)

## Product Decisions I Made (Why This Is a PO Project)
| Decision | What I chose | Why |
|----------|-------------|-----|
| Chunk size | 600 tokens with 100 overlap | Sweet spot for policy docs — precise retrieval without losing context |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Free, fast, no API key required, excellent for semantic search |
| Chunks retrieved | k=5 | More chunks = better recall for multi-section regulatory topics |
| Guardrail | "I don't know" response | Prevents hallucination — bot only answers from document |
| Vector store | FAISS (local) | Portfolio demo doesn't need managed cloud vector store |
| LLM | Anthropic Claude API | Best instruction-following for grounded, cited answers |

## Acceptance Criteria (Written Before Coding)
- [ ] Given a question is asked, When submitted, Then answer appears in < 15 seconds
- [ ] Given an answer is generated, Then it cites the source page number
- [ ] Given a question outside document scope, Then bot says "I don't have enough information"
- [ ] Given the app starts, Then document is indexed automatically on first run
- [ ] Given the index exists, Then it loads from cache without re-embedding

## RAG Architecture

## Tech Stack
- **UI:** Streamlit (Python)
- **RAG Framework:** LangChain
- **Embeddings:** HuggingFace all-MiniLM-L6-v2 (free, local)
- **Vector Store:** FAISS
- **LLM:** Anthropic Claude API
- **Deployment:** Streamlit Community Cloud

## Run Locally
```bash
pip install -r requirements.txt
# Add ANTHROPIC_API_KEY to .streamlit/secrets.toml
python -m streamlit run app.py
```

## About
Built by [Padmini Nagarajan](https://www.linkedin.com/in/padmini-nagarajan/), 
Product Owner with 10+ years in data analytics and digital workflow transformation. 
This project is part of my AI Product Owner portfolio.
