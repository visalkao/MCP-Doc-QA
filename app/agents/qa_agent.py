import openai
from app.config import settings
from app.utils.embeddings import embed_texts

openai.api_key = settings.OPENAI_API_KEY

# Each excerpt is followed by its citation in parens: (source — page X — para Y).

PROMPT_TEMPLATE = """
You are a helpful assistant answering a user's question using only the provided excerpts. 
When you answer, include a short list of citations used, and for any factual claim include the citation.

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

def answer_query(query: str, faiss_index, top_k: int = 5):
    qvec = embed_texts([query])
    results = faiss_index.search(qvec.astype('float32'), k=top_k)[0]
    snippets = []
    for meta, score in results:
        snippets.append({'text': meta['text'], 'metadata': meta, 'score': score})
    context = build_context_snippets(snippets)
    prompt = PROMPT_TEMPLATE.format(context=context, question=query)


    # resp = openai.chat.completions.create(
    #     model=settings.LLM_MODEL,
    #     messages=[{"role": "user", "content": prompt}],
    #     max_tokens=512,
    #     temperature=0.0,
    # )
    resp = openai.ChatCompletion.create(
        model=settings.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.0,
    )
    answer = resp['choices'][0]['message']['content']

    answer = resp.choices[0].message.content

    return {'answer': answer, 'sources': [s['metadata'] for s in snippets]}
