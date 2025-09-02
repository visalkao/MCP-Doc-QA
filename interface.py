import gradio as gr
import requests
import uuid
from app.utils.faiss_store import FaissStore
import os

# MCP server URL
URL = "http://localhost:8080"

# FAISS index path
INDEX_PATH = "./data/index/faiss.idx"

# Reset index at the start
def reset_index():
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
    store = FaissStore(dim=384, index_path=INDEX_PATH)
    store.save()
    return store

reset_index()
print("Index reset")

# Conversation history per session: {session_id: [(user, bot), ...]}
conversation_history = {}

# Uploaded files per session (so user uploads only once)
uploaded_files = {}

# Upload function (only for the first time)
def upload_pdf(file, session_id):
    if session_id not in uploaded_files:
        uploaded_files[session_id] = []
    if file is None:
        return "No file uploaded."
    
    try:
        with open(file.name, "rb") as f:
            r = requests.post(URL + "/upload", files={"file": (file.name, f, "application/pdf")})
        uploaded_files[session_id].append(file.name)
        return f"Uploaded: {file.name}"
    except Exception as e:
        return f"Error uploading file: {str(e)}"

# Chat function
def chat(message, session_id):
    if session_id not in conversation_history:
        conversation_history[session_id] = []

    # Build context from history
    history = conversation_history[session_id]
    context = "\n".join([f"User: {q}\nBot: {a}" for q, a in history])
    full_prompt = f"{context}\nUser: {message}\nBot:" if context else message

    # Query MCP server
    try:
        r = requests.get(URL + "/query", params={"q": full_prompt})
        data = r.json()
        answer = data.get("answer", "No answer returned")
        conversation_history[session_id].append((message, answer))
        return conversation_history[session_id]
    except Exception as e:
        return conversation_history[session_id] + [(message, f"Error: {str(e)}")]

# Build the Gradio chat interface
with gr.Blocks() as demo:
    gr.Markdown("## MCP Document Chatbot")

    # Create a session ID
    session_id = gr.State(str(uuid.uuid4()))

    with gr.Row():
        pdf_file = gr.File(label="Upload PDF (once per session)", file_types=[".pdf"])
        upload_btn = gr.Button("Upload PDF")
        upload_status = gr.Textbox(label="Upload Status", interactive=False)

    chatbox = gr.Chatbot(label="Conversation")
    user_input = gr.Textbox(label="Type your message here...", placeholder="Ask a question...")
    send_btn = gr.Button("Send")

    # Upload PDF click
    upload_btn.click(upload_pdf, inputs=[pdf_file, session_id], outputs=upload_status)

    # Chat click
    send_btn.click(chat, inputs=[user_input, session_id], outputs=chatbox)
    send_btn.click(lambda _: "", inputs=None, outputs=user_input)  # clear input after send

demo.launch()
