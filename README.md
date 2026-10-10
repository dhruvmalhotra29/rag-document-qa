# RAG Document QA

A production-oriented Retrieval-Augmented Generation (RAG) application for question answering over uploaded PDF documents.

The system combines **semantic retrieval, keyword retrieval, Reciprocal Rank Fusion (RRF), and Cross-Encoder reranking** to retrieve relevant document context before generating answers with Gemini.

## Features

* PDF document upload and text extraction
* Configurable text chunking
* Semantic retrieval with FAISS
* Keyword retrieval with BM25
* Hybrid retrieval using Reciprocal Rank Fusion (RRF)
* Cross-Encoder reranking
* Top-3 context selection for answer generation
* Gemini-powered question answering
* Source document and page attribution
* Retrieval evaluation using Recall@3

## Architecture

```text
PDF
 │
 ▼
Text Extraction
 │
 ▼
Chunking
 │
 ├───────────────┐
 ▼               ▼
Embeddings       Text
 │               │
 ▼               ▼
FAISS           BM25
 │               │
 └───────┬───────┘
         ▼
        RRF
         │
         ▼
 Candidate Chunks
         │
         ▼
 Cross-Encoder
  Reranking
         │
         ▼
   Top-3 Chunks
         │
         ▼
       Gemini
         │
         ▼
    Final Answer
         │
         ▼
 Source + Page
```

## Retrieval Pipeline

### Semantic Retrieval

Document chunks are converted into embeddings using:

`all-MiniLM-L6-v2`

FAISS performs vector similarity search to retrieve semantically relevant chunks.

### Keyword Retrieval

BM25 performs lexical retrieval over the original chunk text. This helps preserve exact terms, phrases, numbers, and other keyword-sensitive information.

### Hybrid Retrieval

Results from FAISS and BM25 are combined using **Reciprocal Rank Fusion (RRF)** to produce a unified candidate ranking.

### Reranking

The highest-ranked candidates are passed to a Cross-Encoder:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

The Cross-Encoder evaluates the relationship between the user's query and each candidate chunk and reranks them based on relevance.

The final **top 3 chunks** are provided to Gemini as context.

## Evaluation

The retrieval pipeline is evaluated using **Recall@3**.

Current evaluation on the project's test set:

**Recall@3: 100% (5/5)**

Recall@3 measures whether the relevant chunk is present within the top 3 retrieved results.

## Tech Stack

**Backend**

* Python
* FastAPI
* PyMuPDF
* Sentence Transformers
* FAISS
* BM25
* Cross-Encoder
* Gemini API

**Frontend**

* React
* Vite
* Axios

## Local Development

### Backend

```bash
cd backend

python -m venv venv
```

Activate the virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key
```

Start the API:

```bash
uvicorn src.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Limitations

* Current vector storage is in-memory and is not persistent across application restarts.
* The current application is designed for single-user interaction rather than multi-user document isolation.
* PDF is currently the supported document format.

## Future Improvements

* Persistent vector storage using PostgreSQL/pgvector
* Improved document and metadata management
* Additional document format support
* More comprehensive retrieval evaluation
* Production deployment and monitoring


## Demo

- **Project Walkthrough Video:** [Watch Demo](https://drive.google.com/file/d/1D0GrCAirqs4NiPPayAly64q8CwLLn1si/view?usp=drive_link)
- **Live Application:** [Try RAG Document Assistant](https://gen-lang-client-0567884138.web.app/)

---

## Author

- [@dhruvmalhotra29](https://github.com/dhruvmalhotra29)