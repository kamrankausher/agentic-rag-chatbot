# Agentic AI RAG Chatbot

This project is developed as part of the AI Engineer Intern assessment for Appening Infotech.

The chatbot answers questions only from the provided Agentic AI eBook using Retrieval Augmented Generation (RAG).

Knowledge Base

https://konverge.ai/pdf/Ebook-Agentic-AI.pdf

---

## Features

- PDF ingestion
- Automatic text chunking
- Text embeddings using Sentence Transformers
- ChromaDB vector database
- Retrieval using semantic search
- LangGraph workflow
- Google Gemini for answer generation
- FastAPI REST API
- Grounded responses (No hallucination)
- Confidence score
- Retrieved context chunks

---

## Project Structure

```
agentic-rag-chatbot
│
├── app
│   ├── api.py
│   ├── config.py
│   ├── embeddings.py
│   ├── graph.py
│   ├── ingest.py
│   ├── prompts.py
│   ├── rag.py
│   └── retriever.py
│
├── data
│   └── agentic_ai_ebook.pdf
│
├── chroma_db
│
├── index.py
├── main.py
├── requirements.txt
├── README.md
└── sample_queries.md
```

---

## Architecture

```
                PDF

                 │

          PDF Ingestion

                 │

           Text Chunking

                 │

         Text Embeddings

                 │

             ChromaDB

                 │

         Similarity Search

                 │

            LangGraph

      Retrieve → Generate

                 │

        Google Gemini LLM

                 │

           FastAPI API

                 │

             Response
```

---

## Technologies Used

- Python
- FastAPI
- LangGraph
- ChromaDB
- Google Gemini
- Sentence Transformers
- LangChain
- PyMuPDF

---

## Installation

Clone the repository

```bash
git clone https://github.com/kamrankausher/agentic-rag-chatbot.git
```

Go to project folder

```bash
cd agentic-rag-chatbot
```

Create virtual environment

```bash
python -m venv .venv
```

Activate environment

Windows

```bash
.venv\Scripts\activate
```

Linux/Mac

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variable

Create a `.env` file.

```
GOOGLE_API_KEY=YOUR_API_KEY
```

---

## Index the PDF

Run

```bash
python index.py
```

This will

- Read the PDF
- Split into chunks
- Generate embeddings
- Store vectors in ChromaDB

---

## Start API

```bash
python main.py
```

API will start on

```
http://127.0.0.1:8000
```

Swagger Documentation

```
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### GET /

Returns project information.

---

### GET /health

Returns API health status.

---

### POST /chat

Request

```json
{
    "question":"What is Agentic AI?"
}
```

Response

```json
{
    "question":"What is Agentic AI?",
    "answer":"...",
    "confidence":0.58,
    "retrieved_chunks":[]
}
```

---

## Sample Questions

See

```
sample_queries.md
```

---

## How It Works

1. Read the PDF
2. Split the text into chunks
3. Generate embeddings
4. Store embeddings in ChromaDB
5. User asks a question
6. Similar chunks are retrieved
7. LangGraph creates the workflow
8. Gemini generates an answer using only the retrieved context
9. API returns the answer with retrieved chunks and confidence score

---

## Future Improvements

Some improvements that can be added later are

- Better chunking strategy
- Hybrid Search
- Re-ranking
- Conversation memory
- Streamlit UI
- Docker deployment

---

## Author

Kamran Kausher

B.Tech Computer Science and Engineering
