import os
from dotenv import load_dotenv
from google import genai

load_dotenv()  # reads .env into environment variables
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 1. Text generation
response = client.models.generate_content(
    model=os.getenv("CHAT_MODEL"),
    contents="Explain RAG in one sentence."
)
print("ANSWER:", response.text)

# 2. Embedding
result = client.models.embed_content(
    model=os.getenv("EMBED_MODEL"),
    contents="What is the course fee?"
)
vector = result.embeddings[0].values
print("Embedding length:", len(vector))
print("First 5 numbers:", vector[:5])