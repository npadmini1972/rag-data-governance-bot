import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

def get_embeddings():
    """
    PO Decision: Using HuggingFace free embeddings instead of OpenAI.
    'all-MiniLM-L6-v2' is fast, lightweight, and excellent for 
    semantic search on policy documents. No API key required.
    """
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

def ingest_document(pdf_path: str):
    print(f"Loading document: {pdf_path}")
    
    # Step 1 — Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    
    # Step 2 — Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    # Step 3 — Embed and store
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Step 4 — Save index locally
    vectorstore.save_local("faiss_index")
    print("✅ Index saved to faiss_index/")
    
    return vectorstore

def load_vectorstore():
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore