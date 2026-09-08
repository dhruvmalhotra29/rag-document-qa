from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from fastapi.middleware.cors import CORSMiddleware
from src.services.text_chunker import chunker_text
from src.services.embedding import generate_embeddings
from src.services.vector_store import VectorStore
from src.services.llm import generate_answer, generate_retrieval_query
from dotenv import load_dotenv
from pydantic import BaseModel
from pathlib import Path
import fitz
import os

app = FastAPI()

limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit reached. Please try again after 1 hour."
        }
    )

class QueryRequest(BaseModel):
    question: str
    history: list = []

load_dotenv()

allowed_origins = os.getenv(
    "ALLOWED_ORIGINS"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

vector_store = VectorStore(dimension=384)

@app.get("/")
def home():
    return {"message" : "AI Document Assistant API Running"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    vector_store.clear()
    
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    document = fitz.open(file_path)

    text = ""
    page_chunks = []

    for page_number, page in enumerate(document):
        text = page.get_text()

        chunks = chunker_text(text)

        for chunk in chunks:
            page_chunks.append({
                "text":chunk,
                "page": page_number + 1,
                "filename": file.filename
            })

    #print("PAGE CHUNKS:", page_chunks)
    #print("NUMBER OF CHUNKS:", len(page_chunks))

    if not page_chunks or not any(chunk["text"].strip() for chunk in page_chunks):
        raise HTTPException(
            status_code=400,
            detail="The PDF contains no extractable text."
        )

    texts = [doc["text"] for doc in page_chunks]

    embeddings = generate_embeddings(texts)

    vector_store.add_embeddings(embeddings, page_chunks)

    # print(embeddings[0])

    return {
        "message" : "File uploaded successfully",
        "filename" : file.filename,
        "text": text,
        "total_chunks": len(chunks),
        "first_chunks": chunks[0],
        "embeddings shape": embeddings.shape
    }

@app.post("/query")
@limiter.limit("10/hour")
async def query_document(request: Request, query: QueryRequest):
    question = query.question
    history = query.history

    if not question.strip():
        return {
            "question": question,
            "answer": "Please enter a question.",
            "sources": []
        }

    history_text = "\n".join(
        f"User: {msg['user']}\nAI: {msg['ai']}"
        for msg in history
    )

    retrieval_query = generate_retrieval_query(question,history)
    #print("Retrieval Query", retrieval_query)
    query_embedding = generate_embeddings([retrieval_query])
    
    # Compare FAISS and Hybrid
    faiss_results = vector_store.search(
        query_embedding,
        retrieval_query,
        k=3,
        mode="faiss"
    )

    hybrid_results = vector_store.search(
        query_embedding,
        retrieval_query,
        k=3,
        mode="hybrid"
    )

    #print("\n" + "=" * 60)
    #print("QUESTION:", question)
    #print("RETRIEVAL QUERY:", retrieval_query)
    #print("=" * 60)

    #print("\n--- FAISS RESULTS ---")
    #for i, result in enumerate(faiss_results, 1):
    #    print(f"\n{i}. {result['text'][:500]}")
    #    print("Page:", result["page"])

    #print("\n--- HYBRID RESULTS ---")
    #for i, result in enumerate(hybrid_results, 1):
    #    print(f"\n{i}. {result['text'][:500]}")
    #    print("Page:", result["page"])

    # Use Hybrid for the actual RAG answer
    results = hybrid_results

    if not results:
        return {
            "question": question,
            "answer": "I couldn't find relevant information in the uploaded document."
        }

    context = "\n\n".join(result["text"] for result in results)

    answer, answer_found = generate_answer(question, context, history_text)


    sources = []

    if answer_found:
        for result in results:
            source = {
                "filename": result["filename"],
                "page": result["page"]
            }

            if source not in sources:
                sources.append(source)

    return {
        "question": question,
        "answer": answer,
         "sources": sources
        #"relevant_chunks": results
    }

@app.get("/evaluate")
async def evaluate():

    test_cases = [
        {
            "question": "What is the annual leave entitlement?",
            "expected_keyword": "20 days"
        },
        {
            "question": "Can annual leave be carried forward?",
            "expected_keyword": "carried forward"
        },
        {
            "question": "When does the laptop need to be returned?",
            "expected_keyword": "return it upon leaving"
        },
        {
            "question": "What is the annual training budget?",
            "expected_keyword": "$1,000"
        },
        {
            "question": "How many days per week can employees work remotely?",
            "expected_keyword": "three days per week"
        }
    ]

    correct = 0
    k = 3

    #print("\n" + "=" * 60)
    #print("RAG RETRIEVAL EVALUATION")
    #print("=" * 60)

    for test in test_cases:

        question = test["question"]
        expected = test["expected_keyword"]

        query_embedding = generate_embeddings(
            [question]
        )

        results = vector_store.search(
            query_embedding,
            question,
            k=k,
            mode="hybrid"
        )

        found = False

    #    print("\nQuestion:", question)

        for result in results:

            text = result["text"].lower()

            if expected.lower() in text:
                found = True

    #        print(
    #            "Page:",
    #            result["page"],
    #            "| Match:",
    #            expected.lower() in text
    #        )

        if found:
            correct += 1

    #    print(
    #        "Found:",
    #        "YES ✅" if found else "NO ❌"
    #    )

    recall = correct / len(test_cases)

    #print("\n" + "=" * 60)
    #print(f"Recall@{k}: {recall:.2%}")
    #print("=" * 60)

    return {
        "recall_at_3": recall,
        "correct": correct,
        "total": len(test_cases)
    }