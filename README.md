# ALFRED — Autonomous Research Assistant

ALFRED is a personal autonomous research assistant built to help users with queries, provide accurate and relevant information, and perform tasks through natural text-based conversations. It is designed to be a reliable, trustworthy source of information — striving for accuracy, neutrality, and professionalism in every interaction.

---

## What ALFRED Can Do

- **Live web search** — searches the internet in real time via DuckDuckGo
- **Wikipedia lookup** — retrieves verified academic and historical information
- **Personal knowledge base** — ingests your own notes, PDFs, and presentations and answers questions from them using semantic search
- **Persistent memory** — remembers conversation history across sessions via a local SQLite database
- **File saving** — saves research outputs to a local text file on request
- **SFT data logging** — automatically logs every conversation turn in ChatML format for future fine-tuning

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent framework | LangGraph |
| LLM orchestration | LangChain |
| Language model | Llama 3.3 70B via OpenRouter |
| Vector database | ChromaDB |
| Embeddings | HuggingFace sentence-transformers (local, no API cost) |
| Persistent memory | SQLite via LangGraph checkpointing |
| Web search | DuckDuckGo Search (ddgs) |
| Document loaders | PyMuPDF, Unstructured, Python-PPTX |

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/MokshNair/ALFRED.git
cd ALFRED
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up your API key**

Create a `.env` file in the root directory:
```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Get your free API key at [openrouter.ai](https://openrouter.ai)

**4. Run ALFRED**
```bash
python main.py
```

---

## Using the Knowledge Base

ALFRED can search your personal documents. To use this feature, drop your files into the `knowledge_base/` folder before running:

```
knowledge_base/
├── notes/        ← .txt and .md files
├── pdfs/         ← .pdf files
└── presentations/ ← .pptx files
```

ALFRED automatically detects new or modified files on startup and indexes them into the vector database. No manual steps required.

---

## Project Structure

```
ALFRED/
├── main.py                  # Entry point — agent loop, SFT logger, file saving
├── memory.py                # RAG pipeline — document ingestion, ChromaDB, embeddings
├── tools.py                 # Tool definitions — web search, Wikipedia, knowledge base
├── requirements.txt         # Pinned dependencies
├── .env                     # API keys (not committed)
├── .env.example             # Template for required environment variables
├── knowledge_base/          # Drop your documents here (not committed)
└── alfred_knowledge_db/     # ChromaDB vector store (auto-generated, not committed)
```

---

## Architecture Decisions

- **Migrated from AgentExecutor to LangGraph** — production-standard agent architecture with proper state management
- **Deterministic file saving** — replaced LLM-controlled file saving with native Python logic to eliminate hallucination bugs
- **Local embeddings** — HuggingFace sentence-transformers run entirely on device, zero API cost
- **Hash-based change detection** — knowledge base only re-indexes new or modified files, not the entire folder on every run
- **Automated SFT logging** — every conversation is logged in ChatML format, passively building a fine-tuning dataset from real usage

---

## Status

ALFRED is an active work in progress. Current focus: adding an evals layer and expanding tool capabilities.
