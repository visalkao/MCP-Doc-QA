from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
from app.config import settings
from app.utils.faiss_store import FaissStore
from app.agents.ocr_agent import ingest_pdf
from app.agents.qa_agent import get_all_snippets, build_context_snippets
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

all_snippets = get_all_snippets(store)
all_context = build_context_snippets(all_snippets) 
print(f"Loaded {len(all_snippets)} snippets for full document context.")

SUMMARIZED_TEXTS = ""

SUMMARIZE_PROMPT_TEMPLATE = """
You are an expert assistant that reads PDF documents and produces a clear, structured, and detailed summary.

Context:
{context}

Instructions:
1. Analyze the entire context carefully. 
2. Write a summary that captures:
   - Main purpose of the document
   - Key sections and their contents
   - Important data points, facts, or arguments
   - Any conclusions, recommendations, or decisions
3. Use clear language that is faithful to the source text.
4. Be detailed but concise: include all essential ideas without copying full paragraphs.
5. Present the summary in structured form with bullet points or short sections.

Final Answer:
"""

@app.post('/upload')
async def upload_pdf(file: UploadFile = File(...)):
    global SUMMARIZED_TEXTS
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, 'Only PDF supported in this endpoint')

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    target = os.path.join(settings.UPLOAD_DIR, file.filename)
    
    # Save uploaded file
    with open(target, 'wb') as f:
        content = await file.read()
        f.write(content)

    n = ingest_pdf(target, store)
    store.save()

    # Run summarization
    prompt = SUMMARIZE_PROMPT_TEMPLATE.format(context=all_context[:15000])  
    # limit context length to avoid token overflow (adjust as needed)

    try:
        resp = openai.ChatCompletion.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3,
        )
        SUMMARIZED_TEXTS = resp.choices[0].message.content.strip()
    except Exception as e:
        print("Summarization failed:", e)
        SUMMARIZED_TEXTS = ""

    return JSONResponse({'ingested_paragraphs': n})
def classify_intent(query: str) -> str:
    """
    Decide whether to answer using:
      - 'chat': no document context
      - 'pdf_rag': targeted PDF vector search (QA style)
      - 'pdf_full': full document context (summarization, themes, whole doc tasks)
    """
    global SUMMARIZED_TEXTS

    prompt = f"""
You are an assistant that decides how to answer a user query.

You are given:
- A short summary of the PDF content
- The user's question

PDF Summary:
{SUMMARIZED_TEXTS if SUMMARIZED_TEXTS else "No summary available"}

Question:
{query}

Options:
- 'chat': The question does not require the PDF at all.
- 'pdf_rag': The question requires consulting specific parts of the PDF (e.g. facts, details, page references).
- 'pdf_full': The question asks for a summary, overview, or holistic analysis of the entire PDF.

Answer with only one of: chat, pdf_rag, or pdf_full
    """
    try:
        resp = openai.ChatCompletion.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        decision = resp.choices[0].message.content.strip().lower()
        print("Decision = ", decision)
        if decision not in ["chat", "pdf_rag", "pdf_full"]:
            return "chat"
        return decision
    except Exception:
        # fallback heuristic if API call fails
        q = query.lower()
        if any(k in q for k in ["summary", "summarize", "overview", "whole document", "themes", "key points"]):
            return "pdf_full"
        elif any(k in q for k in ["invoice", "report", "page", "document", "paragraph"]):
            return "pdf_rag"
        # if we have a summary, default to pdf_rag (since most queries will be about PDF)
        return "pdf_rag" if SUMMARIZED_TEXTS else "chat"

@app.get('/query')
async def query(q: str):
    try:
        decision = classify_intent(q)
        print("DEcision =", decision)
        if decision == "chat":
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
        elif decision == "pdf_rag":
            # 'pdf_rag': use vector searches to find the best match
            from app.agents.qa_agent import answer_query
            res = answer_query(q, store, top_k=settings.TOP_K)
            return res
        else: 
            # 'pdf_full': use all snippets for full context
            from app.agents.qa_agent import answer_query
            res = answer_query(q, store, top_k=settings.TOP_K, use_rag=False)
            res = {'answer': res["answer"], 'sources': []}
            return res
    except Exception as e:
        return {'answer': f'Error retrieving full context: {str(e)}', 'sources': []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)
