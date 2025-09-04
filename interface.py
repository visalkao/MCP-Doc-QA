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

def handle_pdf(file):
    # file is a tempfile object
    return f"Uploaded"

# Build the Gradio chat interface
# with gr.Blocks() as demo:
#     gr.Markdown("## MCP Document Chatbot")

#     # Create a session ID
#     session_id = gr.State(str(uuid.uuid4()))

#     with gr.Row():
#         pdf_file = gr.File(label="Upload PDF (once per session)", file_types=[".pdf"])
#         upload_btn = gr.Button("Upload PDF")
#         upload_status = gr.Textbox(label="Upload Status", interactive=False)

#     chatbox = gr.Chatbot(label="Conversation")
#     with gr.Row():  # horizontal layout
#         with gr.Column(scale=9):  # 90%
#             user_input = gr.Textbox(
#                 placeholder="Ask a question...",
#                 show_label=False
#             )
#             height=90
            
#         with gr.Column(scale=0.5):
#             # pdf_file = gr.File(label="Upload PDF (once per session)", file_types=[".pdf"])
#             # upload_btn = gr.Button("Upload PDF")
#             # # upload_status = gr.Textbox(label="Upload Status", interactive=False)
#             pdf_file = gr.File(
#                 label="",  # hide label
#                 file_types=[".pdf"],
#                 type="binary",  # return file object
#             )    
#             # auto-call function when file is selected
#             pdf_file.upload(handle_pdf, inputs=pdf_file, outputs=None)


#         with gr.Column(scale=0.5):  # 10%
#             send_btn = gr.Button("Send")
#             # send_btn.style(full_width=True, height=user_input.style()["height"])
#             height=90



#     # Upload PDF click
#     upload_btn.click(upload_pdf, inputs=[pdf_file, session_id], outputs=upload_status)

#     # Chat click
#     send_btn.click(chat, inputs=[user_input, session_id], outputs=chatbox)
#     send_btn.click(lambda _: "", inputs=None, outputs=user_input)  # clear input after send

import gradio as gr
import uuid

with gr.Blocks(css="""
               
/* Chat messages */
div[class*="chatbot_message"] {
    font-size: 24px !important;
}

/* Chat input textbox */
textarea {
    font-size: 24px !important;
}

/* Buttons */
button {
    font-size: 24px !important;
}

/* File input label (upload button) */
input[type="file"]::file-selector-button {
    font-size: 24px !important;
}

/* Markdown headings/text */
h1, h2, h3, h4, h5, h6, p {
    font-size: 24px !important;
}

/* Make the container full viewport */
.gradio-container {
    height: 100vh;
    display: flex;
    flex-direction: column;
}

/* Chatbot area takes 90% height */
div[class*="chatbot"] {
    flex: 9 !important;
    overflow-y: auto;
}

/* Input row takes 10% height */
div[class*="row"]:has(textarea), div[class*="row"]:has(button) {
    flex: 1 !important;
    display: flex;
    align-items: center;
}

               /* Style file input as button */
input[type="file"] {
    display: none; /* hide the ugly file input */
}

input[type="file"]::file-selector-button {
    background-color: #007bff; /* Bootstrap blue */
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 16px;
}

input[type="file"]::file-selector-button:hover {
    background-color: #0056b3; /* darker blue */
}

               
/* Make input and buttons full height */
textarea, button, input[type="file"]::file-selector-button {
    height: 100% !important;
}
""") as demo:
    gr.Markdown("## AI Agents Document Chatbot")

    # Session ID
    session_id = gr.State(str(uuid.uuid4()))

    chatbox = gr.Chatbot(label="Conversation")

    with gr.Row():  # single horizontal row for input
        # Textbox column (90%)
        with gr.Column(scale=9):
            user_input = gr.Textbox(
                placeholder="Ask a question...",
                show_label=False
            )

        # File upload column (5%)
        # with gr.Column(scale=0.2):
        #     # pdf_file = gr.File(label="Upload PDF (once per session)", file_types=[".pdf"])
        #     # pdf_status = gr.Textbox(label="", interactive=False)  # optional status
        #     # pdf_file.upload(upload_pdf, inputs=[pdf_file, session_id], outputs=pdf_status)
        #     pdf_file = gr.File(label="", file_types=[".pdf"], visible=False)

        #     # Single Upload button
        #     upload_btn = gr.Button("Upload PDF")
        #     # When button clicked, trigger hidden file input
        #     upload_btn.click(lambda _: pdf_file, inputs=None, outputs=pdf_file)

        #     # Auto-upload handler after file selection
        #     pdf_file.upload(upload_pdf, inputs=[pdf_file, session_id])
        with gr.Column(scale=0.2):
            pdf_file = gr.File(
                label="", 
                file_types=[".pdf"], 
            )

            # Apply CSS later to make it look like a button
            pdf_file.upload(upload_pdf, inputs=[pdf_file, session_id])



        # Send button column (5%)
        with gr.Column(scale=0.8):
            send_btn = gr.Button("Send", variant="primary")

    # Chat click
    send_btn.click(chat, inputs=[user_input, session_id], outputs=chatbox)
    send_btn.click(lambda _: "", inputs=None, outputs=user_input)  # clear input
    # user_input.submit(fn=lambda x: x, inputs=user_input, outputs=user_input)
    user_input.submit(chat, inputs=[user_input, session_id], outputs=chatbox)
    user_input.submit(lambda _: "", inputs=None, outputs=user_input) 

demo.launch()

