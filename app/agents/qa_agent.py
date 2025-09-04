import openai
from app.config import settings
from app.utils.embeddings import embed_texts
from app.utils.faiss_store import FaissStore
openai.api_key = settings.OPENAI_API_KEY


PROMPT_TEMPLATE = """
You are a helpful assistant answering using the excerpts when relevant. Make sure all of your answers are correct and if you find the context provided is not enough, you can refuse to answer due to lack of material. When you answer, include a short list of citations used, and for any factual claim include the citation.
But if the user requests a summary or overall analysis, synthesize across all available context.

Context:
{context}

Question: {question}

Answer:
"""

def build_context_snippets(snippets: list) -> str:
    lines = []
    for s in snippets:
        md = s['metadata']
        lines.append(f"[{md['source']} — page {md['page']} — para {md['paragraph_id']}] {s['text']}")
    return "\n\n".join(lines)

def get_all_snippets(index: FaissStore):
    results = index.get_all()
    return [{'text': meta['text'], 'metadata': meta, 'score': score} for meta, score in results]

def answer_query(query: str, faiss_index, top_k: int = 5, use_rag = True):
    try:
        if use_rag:
            qvec = embed_texts([query])
            results = faiss_index.search(qvec.astype('float32'), k=top_k)[0]
            snippets = [{'text': meta['text'], 'metadata': meta, 'score': score} for meta, score in results]
            context = build_context_snippets(snippets)
        else:
                snippets = get_all_snippets(faiss_index)
                context = build_context_snippets(snippets)  

        prompt = PROMPT_TEMPLATE.format(context=context, question=query)
        resp = openai.ChatCompletion.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        answer = resp['choices'][0]['message']['content']

        answer = resp.choices[0].message.content
    except Exception as e:
        return {'answer': f'Error retrieving full context: {str(e)}', 'sources': []}
    return {'answer': answer, 'sources': [s['metadata'] for s in snippets]}
