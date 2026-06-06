import os
import streamlit as st
from anthropic import Anthropic
from langchain_community.vectorstores import FAISS
from ingest import ingest_document, load_vectorstore

# ─── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="FDA Part 11 Governance Q&A Bot",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Initialize API Clients ─────────────────────────────────────
try:
    anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
    openai_key = st.secrets["OPENAI_API_KEY"]
    client = Anthropic(api_key=anthropic_key)
except Exception:
    st.error("API keys not found. Check .streamlit/secrets.toml")
    st.stop()

# ─── Load or Build Vector Store ─────────────────────────────────
# PO Decision: Build index once, cache it for all users
# This avoids re-embedding on every question (cost + speed)
@st.cache_resource
def get_vectorstore():
    pdf_path = "docs/fda_part11.pdf"
    index_path = "faiss_index"
    
    if os.path.exists(index_path):
        st.sidebar.success("✅ Index loaded from cache")
        return load_vectorstore()
    else:
        st.sidebar.info("⏳ Building index for first time...")
        return ingest_document(pdf_path)

vectorstore = get_vectorstore()

# ─── RAG Answer Function ─────────────────────────────────────────
def get_answer(question: str, chat_history: list) -> tuple:
    """
    PO Decision: Retrieve top 3 chunks, pass as context to Claude.
    Claude answers ONLY from the document — not from general knowledge.
    If the answer isn't in the document, the bot says so.
    This is your guardrail against hallucination.
    """
    # Step 1 — Retrieve relevant chunks
    docs = vectorstore.similarity_search(question, k=5)
    
    # Step 2 — Build context from retrieved chunks
    context = "\n\n---\n\n".join([
        f"[Page {doc.metadata.get('page', 'N/A') + 1}]\n{doc.page_content}"
        for doc in docs
    ])
    
    # Step 3 — Build conversation history for Claude
    messages = chat_history + [{
        "role": "user",
        "content": f"""You are a helpful data governance assistant. 
Answer the question below using ONLY the document context provided.
If the answer is not in the context, say exactly:
"I don't have enough information in this document to answer that question."
Do NOT use any outside knowledge.

Always end your answer with:
📄 Source: Page [X] of the FDA Part 11 guidance document.

Document Context:
{context}

Question: {question}"""
    }]
    
    # Step 4 — Generate answer with Claude
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        messages=messages
    )
    
    answer = response.content[0].text
    
    # Step 5 — Return answer + source pages for citation
    source_pages = list(set([
        doc.metadata.get('page', 0) + 1 for doc in docs
    ]))
    
    return answer, source_pages

# ─── UI Layout ───────────────────────────────────────────────────
st.title("📋 FDA Part 11 Governance Q&A Bot")
st.markdown(
    "*Ask any question about FDA 21 CFR Part 11 — "
    "Electronic Records & Electronic Signatures.*"
)
st.divider()

# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💡 Sample Questions")
    st.markdown("""
- What is the scope of Part 11?
- What are the audit trail requirements?
- What does Part 11 say about electronic signatures?
- What are the legacy system requirements?
- What is enforcement discretion?
- What are the record retention requirements?
- How should copies of records be provided?
- What records require validation?
    """)

# ─── Chat Interface ───────────────────────────────────────────────
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if question := st.chat_input(
    "Ask a question about FDA Part 11 governance..."
):
    # Display user question
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Generate and display answer
    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            try:
                answer, source_pages = get_answer(
                    question,
                    st.session_state.chat_history
                )
                st.markdown(answer)
                
                # Update conversation history for Claude
                st.session_state.chat_history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
                
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

# ─── Footer ──────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built by Padmini Nagarajan │ AI Product Owner Portfolio │ "
    "RAG powered by LangChain + Claude + OpenAI Embeddings"
)