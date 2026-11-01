import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

import rag

DEFAULT_PDF = "brightpath_faq.pdf"

# In-memory store for the currently loaded document
store = {"name": None, "chunks": [], "index": None}


def index_pdf(path, name):
    """Read, chunk and embed a PDF, then save it in the store."""
    pages = rag.load_pdf(path)
    if not pages:
        raise HTTPException(400, "No readable text found. The PDF may be a scanned image.")
    chunks = rag.chunk_pages(pages)
    store.update(name=name, chunks=chunks, index=rag.build_index(chunks))
    return len(pages), len(chunks)


@asynccontextmanager
async def lifespan(app):
    """Runs once when the server starts: load the demo PDF so the bot works immediately."""
    if os.path.exists(DEFAULT_PDF):
        index_pdf(DEFAULT_PDF, DEFAULT_PDF)
        print(f"Loaded demo document: {DEFAULT_PDF}")
    yield


app = FastAPI(title="DocuChat API", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def home():
    """Send the chat web page when someone opens the home page."""
    return FileResponse("static/index.html")


@app.get("/status")
def status():
    return {"document": store["name"], "chunks": len(store["chunks"])}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")

    # Save the upload to a temporary file, index it, then delete the temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        pages, chunks = index_pdf(tmp_path, file.filename)
    finally:
        os.remove(tmp_path)

    return {"document": file.filename, "pages": pages, "chunks": chunks}


@app.post("/chat")
def chat(req: ChatRequest):
    if store["index"] is None:
        raise HTTPException(400, "Please upload a PDF first.")
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty.")

    reply, sources = rag.answer(question, store["chunks"], store["index"])
    return {
        "answer": reply,
        "sources": [
            {"page": s["page"], "score": round(s["score"], 3), "text": s["text"][:200]}
            for s in sources
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", reload=True)