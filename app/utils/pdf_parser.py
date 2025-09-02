import pdfplumber
from typing import List, Dict

def pdf_to_pages(path: str) -> List[Dict]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, p in enumerate(pdf.pages, start=1):
            text = p.extract_text() or ""
            pages.append({"page_number": i, "text": text})
    return pages

def extract_paragraphs_from_text(text: str) -> List[str]:
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    out = []
    for p in paras:
        if len(p) > 1000:
            s = p.replace('\n', ' ')
            for chunk in s.split('. '):
                c = chunk.strip()
                if c:
                    out.append(c + ('.' if not c.endswith('.') else ''))
        else:
            out.append(p)
    return out
