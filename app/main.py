from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
from app.config import settings
from app.utils.faiss_store import FaissStore
from app.agents.ocr_agent import ingest_pdf

# Optional: LLM for decision
import openai

app = FastAPI(title='MCP Document Q&A')

from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

_dim = 384
index_path = os.path.join(settings.INDEX_DIR, 'faiss.idx')
store = FaissStore(dim=_dim, index_path=index_path)
try:
    store.load()
except Exception:
    pass


@app.post('/upload')
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, 'Only PDF supported in this endpoint')
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    target = os.path.join(settings.UPLOAD_DIR, file.filename)
    with open(target, 'wb') as f:
        content = await file.read()
        f.write(content)
    n = ingest_pdf(target, store)
    store.save()
    return JSONResponse({'ingested_paragraphs': n})

def classify_intent(query: str) -> str:
    """
    Decide whether to answer using PDF context (FAISS) or just chat.
    Returns 'pdf' or 'chat'.
    """
    # Simple prompt for small LLM
    prompt = f"""
You are an assistant that decides whether a question requires consulting a PDF document or can be answered directly.
Answer with only 'pdf' or 'chat'.

Question: {query}
Answer:
"""
    try:
        resp = openai.ChatCompletion.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        decision = resp.choices[0].message.content.strip().lower()
        if decision not in ["pdf", "chat"]:
            return "chat"
        return decision
    except Exception:
        # fallback to keyword-based
        doc_keywords = ["invoice", "report", "page", "document", "paragraph"]
        return "pdf" if any(k in query.lower() for k in doc_keywords) else "chat"

# @app.get('/query')
# async def query(q: str):
#     try:
#         decision = classify_intent(q)

#         if decision == "chat":
            
#             return {"answer": "Hello! I'm here to help with general questions.", "sources": []}

#         # Otherwise, use FAISS / PDF agent
#         from app.agents.qa_agent import answer_query
#         res = answer_query(q, store, top_k=settings.TOP_K)
#         return res

#     except Exception as e:
#         raise HTTPException(500, str(e))


@app.get('/query')
async def query(q: str):
    try:
        decision = classify_intent(q)

        if decision == "chat":
            # Use OpenAI LLM to answer general questions

            prompt = f"""
You are a helpful assistant. Answer the question as concisely as possible.

Question: {q}
Answer:
"""

            try:
                resp = openai.ChatCompletion.create(
                    model=settings.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.5,
                )
                answer = resp.choices[0].message.content.strip()
            except Exception as e:
                # fallback in case the API fails
                answer = f"Could not generate answer: {str(e)}"

            return {"answer": answer, "sources": []}

        # Otherwise, use FAISS / PDF agent
        from app.agents.qa_agent import answer_query
        res = answer_query(q, store, top_k=settings.TOP_K)
        return res

    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)
