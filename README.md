# MCP — Universal Business Document Q\&A Copilot



https://github.com/user-attachments/assets/9b056f0e-8f92-451b-8397-ee41df072fea



* Watch the detailed demo below

**MCP** is a Universal Business Document Q\&A Copilot that allows users to upload business documents (PDFs, scanned images) and ask natural language questions. The system uses OCR, paragraph segmentation, vector embeddings, and LLMs to provide accurate answers with precise references (page, paragraph).

## Features

* **Document Upload & OCR:** Upload PDFs or images; system extracts text using OCR (Tesseract) or PDF parsers.
* **Paragraph Segmentation:** Breaks down documents into manageable chunks for indexing and retrieval.
* **Vector Search:** Converts paragraphs into embeddings and stores them in FAISS for fast similarity search.
* **LLM-Powered Q\&A:** Uses GPT-4 or another LLM to answer questions based on relevant document paragraphs, with citations.
* **Precise References:** Returns answers with page number, paragraph id, and source file.

## Project Structure

```
mcp-copilot/
├─ app/
│  ├─ main.py                 # FastAPI server + endpoints
│  ├─ agents/                 # OCR, parser, QA agents
│  ├─ utils/                  # PDF parsing, OCR, embeddings, FAISS store
│  └─ config.py               # Config & API keys
├─ data/                      # Uploaded files and index storage
├─ resources/videos/demo1.mp4 # Demo video
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
└─ sample_client.py           # Script to test API
└─ interface.py               # UI design (Gradio)
```

## Installation

```bash
# Clone the repo
git clone <https://github.com/visalkao/MCP-Doc-QA.git>
cd MCP-Doc-QA

# Install dependencies
pip install -r requirements.txt

```

## Usage

You can either run the interface (+server) or use it with bash. 
### Run the interface

#### Server (FastAPI)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

#### UI

```bash
python interface.py
```

Or 

### Bash
#### Upload a document

```bash
curl -F 'file=@invoice.pdf' http://localhost:8080/upload
```

#### Ask a question

```bash
curl 'http://localhost:8080/query?q=What%20is%20the%20invoice%20number?'
```

## Demo

![MCP Demo Video](resources/videos/demo1.mp4)

Watch the demo video above to see MCP in action: document upload, question querying, and precise answer retrieval.

## Configuration

* Update `app/config.py` or `.env` with:

  * `OPENAI_API_KEY` for GPT API access
  * Paths for uploads and index storage
  * LLM and embedding model choices


## License

MIT License
