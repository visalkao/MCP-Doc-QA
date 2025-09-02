from app.utils.pdf_parser import pdf_to_pages, extract_paragraphs_from_text
from app.utils.embeddings import embed_texts
from app.utils.faiss_store import FaissStore
import numpy as np
import os

def ingest_pdf(file_path: str, index: FaissStore):
    pages = pdf_to_pages(file_path)
    all_texts = []
    metadatas = []
    for p in pages:
        paras = extract_paragraphs_from_text(p['text'] or '')
        for pid, para in enumerate(paras, start=1):
            meta = {
                'source': os.path.basename(file_path),
                'page': p['page_number'],
                'paragraph_id': pid,
                'text': para[:800]
            }
            all_texts.append(para)
            metadatas.append(meta)
    if not all_texts:
        return 0
    vectors = embed_texts(all_texts)
    index.add(vectors.astype(np.float32), metadatas)
    return len(all_texts)
