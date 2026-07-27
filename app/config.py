from pathlib import Path
import os

from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

PDF_PATH = DATA_DIR / "agentic_ai_ebook.pdf"

CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# ---------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------

GEMINI_MODEL = "gemini-3.6-flash"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------------------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------------------

CHUNK_SIZE = 500

CHUNK_OVERLAP = 100

# ---------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------

TOP_K = 4

# ---------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------

API_TITLE = "Agentic AI RAG Chatbot"

API_VERSION = "1.0.0"

API_DESCRIPTION = (
    "RAG-based chatbot using LangGraph, ChromaDB and Gemini."
)

HOST = "127.0.0.1"

PORT = 8000