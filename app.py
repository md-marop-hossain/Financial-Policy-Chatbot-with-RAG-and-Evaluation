import os
import numpy as np
import pickle
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import nest_asyncio
from typing import Optional, Dict, List

# ----------------- Setup -----------------
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is missing or not loaded from .env")

client = OpenAI(api_key=api_key)
nest_asyncio.apply()

# Load FAISS index + valid chunks
index = faiss.read_index("index/faiss_index.idx")
with open("index/valid_chunks.pkl", "rb") as f:
    valid_chunks = pickle.load(f)

# ----------------- Embeddings -----------------
def get_embeddings_safe(texts: List[str]):
    texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return []
    try:
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=texts
        )
        return [d.embedding for d in response.data]
    except Exception as e:
        print("Embedding error:", e)
        return []

# ----------------- Retrieval -----------------
def search_similar_chunks(query: str, k: int = 15):
    query_embedding = get_embeddings_safe([query])[0]
    D, I = index.search(np.array([query_embedding]), k)
    return [valid_chunks[i] for i in I[0]]

# ----------------- Prompt Template -----------------
def build_financial_policy_prompt(context: str, query: str) -> str:
    return f"""You are a Financial Policy Chatbot that answers user questions ONLY using the provided context from a financial policy document.

Instructions:
- Answer in English only.
- Use information strictly from the context below; do not invent facts.
- Keep answers clear and helpful (about 1–3 sentences).
- Include page citations in parentheses like (Page 2) or (Page 2, Page 3).
- If the answer is not in the context, reply exactly: Not found in the document.

Context:
{context}

Question: {query}
Answer:"""

# ----------------- Answer Generation -----------------
def generate_answer_from_context(query: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt = build_financial_policy_prompt(context, query)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

# ----------------- Evaluation -----------------
def evaluate_cosine_similarity(query: str):
    query_emb = np.array([get_embeddings_safe([query])[0]])
    chunks = search_similar_chunks(query)
    chunk_embs = [get_embeddings_safe([c])[0] for c in chunks]
    scores = cosine_similarity(query_emb, chunk_embs)[0]
    return {
        "average": round(np.mean(scores), 4),
        "scores": [round(s, 4) for s in scores]
    }

def check_groundedness(query: str):
    answer = generate_answer_from_context(query, search_similar_chunks(query))
    context = "\n\n".join(search_similar_chunks(query))
    prompt = f"""Check whether the answer is grounded strictly in the given context.

Context:
{context}

Question: {query}
Answer: {answer}

Reply YES or NO and explain briefly (English only)."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# ----------------- Conversation Memory -----------------
conversation_memory: Dict[str, List[Dict[str, str]]] = {}
FOLLOW_UP_HINTS = ("what about", "then", "this", "that")

def build_disambiguated_query(session_id: str, current_q: str) -> str:
    history = conversation_memory.get(session_id, [])
    if not history:
        return current_q
    q_lower = current_q.strip().lower()
    if any(hint in q_lower for hint in FOLLOW_UP_HINTS) or len(current_q.split()) <= 4:
        recent = "\n".join(h["q"] for h in history[-2:])
        return f"Previous:\n{recent}\nCurrent:\n{current_q}"
    return current_q

# ----------------- FastAPI App -----------------
app = FastAPI()

class Query(BaseModel):
    question: str
    session_id: Optional[str] = None

@app.get("/")
def root():
    return {"status": "RAG API Running", "endpoints": ["/ask", "/evaluate", "/chat"]}

@app.post("/ask")
def ask_rag(data: Query):
    answer = generate_answer_from_context(data.question, search_similar_chunks(data.question))
    return {
        "question": data.question,
        "answer": answer,
        "top_chunks": search_similar_chunks(data.question)
    }

@app.post("/evaluate")
def evaluate_rag(data: Query):
    return {
        "question": data.question,
        "answer": generate_answer_from_context(data.question, search_similar_chunks(data.question)),
        "cosine_similarity": evaluate_cosine_similarity(data.question),
        "groundedness_check": check_groundedness(data.question)
    }

@app.post("/chat")
def chat_rag(data: Query):
    session = data.session_id or "default"
    history = conversation_memory.get(session, [])

    # -------- Memory-Aware Meta Questions --------
    q_lower = data.question.lower().strip()
    if "previous question" in q_lower:
        if history:
            prev_q = history[-1]["q"]
            answer = f"Your previous question was: \"{prev_q}\""
        else:
            answer = "You haven't asked any previous questions yet."
        history.append({"q": data.question, "a": answer})
        conversation_memory[session] = history[-10:]
        return {
            "question": data.question,
            "answer": answer,
            "top_chunks": [],
            "history": conversation_memory[session]
        }

    if "previous answer" in q_lower:
        if history:
            prev_a = history[-1]["a"]
            answer = f"My previous answer was: \"{prev_a}\""
        else:
            answer = "I haven't given any answers yet."
        history.append({"q": data.question, "a": answer})
        conversation_memory[session] = history[-10:]
        return {
            "question": data.question,
            "answer": answer,
            "top_chunks": [],
            "history": conversation_memory[session]
        }

    # -------- Normal RAG Flow --------
    disamb_query = build_disambiguated_query(session, data.question)
    top_chunks = search_similar_chunks(disamb_query)
    answer = generate_answer_from_context(data.question, top_chunks)

    # Update memory (keep last 10 turns)
    history.append({"q": data.question, "a": answer})
    conversation_memory[session] = history[-10:]

    return {
        "question": data.question,
        "answer": answer,
        "top_chunks": top_chunks,
        "history": conversation_memory[session]
    }
