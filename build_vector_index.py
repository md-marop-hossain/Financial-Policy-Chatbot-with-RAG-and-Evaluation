import os
import numpy as np
import tiktoken
import faiss
import pickle
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is missing or not loaded from .env")

client = OpenAI(api_key=api_key)

input_file = "data/cleaned_text.txt"
os.makedirs("index", exist_ok=True)

def load_text():
    with open(input_file, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, max_tokens=500):
    enc = tiktoken.encoding_for_model("gpt-4")
    words = text.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in words:
        temp = current_chunk + sentence + ". "
        if len(enc.encode(temp)) <= max_tokens:
            current_chunk = temp
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def get_embeddings_safe(texts):
    texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return []

    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return [d.embedding for d in response.data]

def split_text(text, max_tokens=2048):
    enc = tiktoken.encoding_for_model("text-embedding-3-large")
    words = text.split()
    chunks, temp = [], []

    for word in words:
        temp.append(word)
        if len(enc.encode(" ".join(temp))) > max_tokens:
            temp.pop()
            chunks.append(" ".join(temp))
            temp = [word]
    if temp:
        chunks.append(" ".join(temp))
    return chunks

def build_vector_index(text):
    chunks = chunk_text(text)
    print("Total Chunks:", len(chunks))
    enc = tiktoken.encoding_for_model("text-embedding-3-large")
    embeddings, valid_chunks = [], []
    batch, batch_tokens = [], 0
    max_tokens = 8192

    for chunk in chunks:
        tokens = len(enc.encode(chunk))
        if tokens > max_tokens:
            print(f"🔁 Splitting large chunk ({tokens} tokens)")
            for sub in split_text(chunk):
                sub_tokens = len(enc.encode(sub))
                if batch_tokens + sub_tokens > max_tokens:
                    emb = get_embeddings_safe(batch)
                    embeddings.extend(emb)
                    valid_chunks.extend(batch)
                    batch = [sub]
                    batch_tokens = sub_tokens
                else:
                    batch.append(sub)
                    batch_tokens += sub_tokens
        else:
            if batch_tokens + tokens > max_tokens:
                emb = get_embeddings_safe(batch)
                embeddings.extend(emb)
                valid_chunks.extend(batch)
                batch = [chunk]
                batch_tokens = tokens
            else:
                batch.append(chunk)
                batch_tokens += tokens
    if batch:
        emb = get_embeddings_safe(batch)
        embeddings.extend(emb)
        valid_chunks.extend(batch)

    embeddings = np.array(embeddings)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    print("✅ Embeddings created for", len(valid_chunks), "chunks.")
    faiss.write_index(index, "index/faiss_index.idx")
    with open("index/valid_chunks.pkl", "wb") as f:
        pickle.dump(valid_chunks, f)

    print("✅ Vector index and chunks saved.")

if __name__ == "__main__":
    text = load_text()
    build_vector_index(text)
