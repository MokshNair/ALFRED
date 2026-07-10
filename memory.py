import os
import json
import hashlib
from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Core Paths
CHROMA_PATH = "alfred_knowledge_db"
KNOWLEDGE_PATH = "knowledge_base"
STATE_FILE = "knowledge_base_state.json"

# Local Embedding Model
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_file_hash(filepath: str) -> str:
    #Generates an MD5 hash of a file to detect changes.
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def sync_knowledge_base():
    #Scans for new/modified files, updates ChromaDB, and returns the vector store.
    if not os.path.exists(KNOWLEDGE_PATH):
        os.makedirs(os.path.join(KNOWLEDGE_PATH, "notes"))
        os.makedirs(os.path.join(KNOWLEDGE_PATH, "pdfs"))
        os.makedirs(os.path.join(KNOWLEDGE_PATH, "presentations"))
        print(f"[MEMORY] Created knowledge_base directories. Drop files in there.")

    # Load ledger of previously processed files
    processed_files = {}
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            processed_files = json.load(f)

    new_docs = []
    
    # Walk through all directories in knowledge_base
    for root, _, files in os.walk(KNOWLEDGE_PATH):
        for file in files:
            # Ignore hidden files (like .DS_Store on Mac)
            if file.startswith('.'):
                continue
                
            filepath = os.path.join(root, file)
            file_hash = get_file_hash(filepath)

            # If file is new or modified
            if filepath not in processed_files or processed_files[filepath] != file_hash:
                print(f"[MEMORY] Processing new/updated file: {file}")
                try:
                    if filepath.endswith('.txt') or filepath.endswith('.md') or filepath.endswith('.py'):
                        loader = TextLoader(filepath)
                    elif filepath.endswith('.pdf'):
                        loader = PyMuPDFLoader(filepath)
                    elif filepath.endswith('.pptx'):
                        loader = UnstructuredPowerPointLoader(filepath)
                    else:
                        continue # Skip unsupported formats

                    new_docs.extend(loader.load())
                    processed_files[filepath] = file_hash
                except Exception as e:
                    print(f"[MEMORY] Error loading {filepath}: {e}")

    # Initialize ChromaDB connection
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDING_MODEL)

    # Process and add new documents if any were found
    if new_docs:
        print(f"[MEMORY] Chunking {len(new_docs)} new document pages/slides...")
        # Baseline fallback for raw text, PDFs, and presentations
        base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Specialized splitter configured for Python syntax boundaries
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, 
            chunk_size=1000, 
            chunk_overlap=150
        )
        
        # Specialized splitter configured for Markdown formatting blocks
        markdown_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN, 
            chunk_size=1000, 
            chunk_overlap=150
        )

        # Route each document through its corresponding semantic splitter
        chunks = []
        for doc in new_docs:
            source_path = doc.metadata.get("source", "").lower()
            
            if source_path.endswith('.py'):
                chunks.extend(python_splitter.split_documents([doc]))
            elif source_path.endswith('.md'):
                chunks.extend(markdown_splitter.split_documents([doc]))
            else:
                chunks.extend(base_splitter.split_documents([doc]))
        
        vectorstore.add_documents(chunks)
        
        # Save updated ledger
        with open(STATE_FILE, 'w') as f:
            json.dump(processed_files, f)
            
        print(f"[MEMORY] Added {len(chunks)} chunks to vector database.")
    else:
        print("[MEMORY] Knowledge base is up to date.")

    return vectorstore

def search_knowledge_base(vectorstore, query: str, k: int = 4) -> str:
    """Searches the database for chunks relevant to the query."""
    if not vectorstore:
        return ""
    
    results = vectorstore.similarity_search(query, k=k)
    if not results:
        return ""
    
    context_parts = []
    for doc in results:
        source = doc.metadata.get("source", "Unknown")
        context_parts.append(f"Source: {source}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(context_parts)