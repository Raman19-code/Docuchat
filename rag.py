import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
EMBED_MODEL = os.getenv("EMBED_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")

SYSTEM_PROMPT = """You are a friendly customer support assistant for a business.
Answer the customer's question using ONLY the context provided.

Rules:
- If the answer is not in the context, reply exactly: "Sorry, I don't have that information. Please contact our office for help."
- Never guess or use outside knowledge.
- Keep answers short and clear (2-3 sentences).
- At the end, cite the page number(s) you used, like (Page 3)."""


def load_pdf(path):
    """Read a PDF and return a list of pages: [{"page": 1, "text": "..."}, ...]"""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_pages(pages, chunk_size=100, overlap=20):
    """Split each page into overlapping word chunks, keeping the page number."""
    chunks = []
    step = chunk_size - overlap
    for p in pages:
        words = p["text"].split()
        for start in range(0, len(words), step):
            piece = words[start:start + chunk_size]
            chunks.append({"page": p["page"], "text": " ".join(piece)})
            if start + chunk_size >= len(words):
                break
    return chunks


def embed_texts(texts, task_type):
    """Send a list of texts to Gemini, get back a 2D array: one vector per text."""
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return np.array([e.values for e in result.embeddings])


def normalize(vectors):
    """Scale vectors to length 1, so dot product = cosine similarity."""
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def build_index(chunks):
    """Embed all chunks once. Returns a (num_chunks x 3072) array."""
    vectors = embed_texts([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    return normalize(vectors)


def search(question, chunks, index, top_k=3):
    """Find the top_k chunks most similar in meaning to the question."""
    q = normalize(embed_texts([question], "RETRIEVAL_QUERY")[0])
    scores = index @ q
    best = np.argsort(scores)[::-1][:top_k]
    return [{**chunks[i], "score": float(scores[i])} for i in best]


def build_prompt(question, results):
    """Combine retrieved chunks and the question into one message."""
    context = "\n\n".join(f"[Page {r['page']}] {r['text']}" for r in results)
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer(question, chunks, index):
    """Full RAG: retrieve relevant chunks, then ask Gemini to answer from them."""
    results = search(question, chunks, index)
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=build_prompt(question, results),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )
    return response.text, results


if __name__ == "__main__":
    pages = load_pdf("brightpath_faq.pdf")
    chunks = chunk_pages(pages)
    index = build_index(chunks)
    print(f"Ready! Indexed {len(chunks)} chunks. Type 'quit' to exit.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue
        reply, sources = answer(q, chunks, index)
        print(f"Bot: {reply}\n")