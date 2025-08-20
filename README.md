# Financial Policy Chatbot with RAG (Retrieval-Augmented Generation) and Evaluation
This project implements an AI-powered Financial Policy Chatbot that allows users to interact with a financial policy document in natural language. The system is designed using Retrieval-Augmented Generation (RAG), ensuring that all chatbot answers are grounded in the provided PDF document and cited with page references. The system uses Tesseract OCR for English text extraction, OpenAI’s ```text-embedding-3-large``` for multilingual embedding, FAISS for semantic retrieval, and ```GPT-4o``` for context-aware responses. A Streamlit interface communicates with a FastAPI backend for smooth and interactive user experience.

The project is structured into four main components:

1. **PDF Text Extraction (```pdf_extractor.py```)**
    - Uses **PyMuPDF** for native PDF text extraction.
    - Falls back to **Tesseract OCR** when dealing with scanned/low-text pages.
    - Cleans and normalizes extracted text.
    - Saves the processed text in ```data/cleaned_text.txt```.

2. **Vector Index Construction (```build_vector_index.py```)**
    - Splits the cleaned text into manageable chunks (≈500 tokens).
    - Generates embeddings using OpenAI’s ```text-embedding-3-large``` model.
    - Stores embeddings in a **FAISS vector index** (```faiss_index.idx```).
    - Maintains a mapping of valid text chunks (```valid_chunks.pkl```).

3. **Backend API (```app.py```)**
    - Built with **FastAPI**, providing three main endpoints:
      - ```/ask``` → retrieves and answers questions using **RAG**.
      - ```/evaluate``` → evaluates answers with cosine similarity and a **groundedness check**.
      - ```/chat``` → supports multi-turn conversations with **session-based memory**.
    - Uses **GPT-4o** to generate concise, English-only answers with strict instructions:
      - Cite policy document pages (e.g., Page 2).
      - Never invent facts — respond *“Not found in the document”* if answer is unavailable.
    - Conversation memory includes **both user questions and bot answers**, enabling follow-ups like *“What about debt?”*.

4. **Frontend User Interface (```rag_ui.py```)**
    - Built with **Streamlit** for an interactive chat experience.
    - Features a **chat-style UI** with styled user/bot messages and timestamps.
    - Supports **session IDs** so users can have separate conversations.
    - Provides an **“Evaluate Answer”** option under each bot response:
      - Displays the **original user question** and the **AI’s answer**.
      - Runs ```/evaluate``` to show:
         - A re-generated grounded answer.
         - Cosine similarity scores (retrieval accuracy).
         - Groundedness verification (Yes/No with explanation).


## 📦 Directory Structure
```
├── app.py                  # FastAPI backend (chat, evaluate, ask endpoints)
├── build_vector_index.py   # Embeds and stores vector data using FAISS
├── pdf_extractor.py        # Extracts and preprocesses text from PDF
├── rag_ui.py               # Streamlit chat interface
│
├── index/                  # Vector database storage
│   ├── faiss_index.idx     # FAISS vector index
│   └── valid_chunks.pkl    # Serialized document chunks
│
├── data/                   # Input & processed documents
│   ├── cleaned_text.txt    # Extracted and cleaned text
│   └── For Task - Policy file.pdf    # Example source PDF (policy file)
│
├── .env                    # API keys (ignored in git)
├── .gitignore              # Git ignore file
└── requirements.txt        # Project dependencies
```

## 🚀 Setup Instructions

**Step 1: Clone the Repository and Set Up Environment**
```
git clone https://github.com/md-marop-hossain/Multilingual-Retrieval-Augmented-Generation-RAG-System.git
cd Multilingual-Retrieval-Augmented-Generation-RAG-System
python -m venv myenv
source myenv/bin/activate  # or myenv\Scripts\activate on Windows
pip install -r requirements.txt
```
**Step 2: Extract Text from PDF**

Run the following command to extract text from the PDF using ```pdf_extractor.py```:
   
```python pdf_extractor.py```

**Step 3: Build Vector Index**

This step includes chunking, embedding, and creating the vector store:

```python build_vector_index.py```

**Step 4: Launch the FastAPI Server**

Start the FastAPI app using:

```uvicorn app:app --reload```

 **Available API Endpoints:**
   - Ask a question : ```POST http://127.0.0.1:8000/ask```
   - Evaluate the system : ```POST http://127.0.0.1:8000/evaluate```
   
**Step 5: Start the Streamlit UI**

```streamlit run rag_ui.py```

## 🛠️ Tools & Libraries Used

| Category              | Tool/Library           | Purpose                                                                 |
|-----------------------|------------------------|-------------------------------------------------------------------------|
| LLM & Embeddings      | OpenAI GPT-4o          | Generates grounded answers from retrieved context                       |
|                       | text-embedding-3-large | Multilingual, high-quality embeddings for semantic chunk comparison     |
| Vector Store          | FAISS                  | Fast Approximate Nearest Neighbor search over dense vector space        |
| Backend Framework     | FastAPI                | Backend API for handling queries, answers, and evaluations              |
| Frontend Interface    | Streamlit              | Provides a user-friendly web UI for interacting with the RAG system     |
| PDF Text Extraction   | PyMuPDF (fitz)         | Extracts formatted Bangla and English text from PDFs                    |
| OCR (Image PDF)       | pytesseract            | Extracts text from scanned or image-based PDFs using Tesseract OCR      |
| Image Processing      | Pillow (PIL)           | Handles image preprocessing for OCR pipeline                            |
| Evaluation            | scikit-learn           | Computes cosine similarity for answer relevance scoring                 |
| API Key Management    | python-dotenv          | Loads sensitive keys like `OPENAI_API_KEY` securely from `.env` file    |
| Async Handling        | nest_asyncio           | Enables async FastAPI in notebook-style or blocking environments        |
| Tokenization Utility  | tiktoken               | Used to count tokens (e.g., for chunk size management)        |
| Data Serialization    | pickle                 | Loads/stores vector index and text chunks                               |
| Math/Numeric          | numpy                  | Vector and matrix operations                                            |
| Typing & Validation   | pydantic               | Defines and validates API input/output models                           |
| Server                | uvicorn                | ASGI server for running FastAPI app                                     |

## 💬 Sample Queries and Outputs

<table>
  <tr>
    <td><img src="images/api.png" width="400"/></td>
    <td><img src="images/question.png" width="400"/></td>
  </tr>
  <tr>
    <td><img src="images/answerOne.png" width="400"/></td>
    <td><img src="images/answerTwo.png" width="400"/></td>
  </tr>
  <tr>
    <td><img src="images/cosn.png" width="400"/></td>
    <td><img src="images/steamlit_interface.png" width="400"/></td>
  </tr>
</table>

## 📑 API Documentation

- Framework: FastAPI

| Endpoint    | Method | Description                                |
|-------------|--------|--------------------------------------------|
| `/`         | GET    | Health check and basic API info            |
| `/ask`      | POST   | Ask a question and get a short answer with relevant context chunks |
| `/evaluate` | POST   | Get answer with evaluation metrics (cosine similarity, groundedness) |

- Input: JSON containing ```"query"``` field (e.g., ```{"query": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?"}```)

- Output: JSON with ```"answer"```, and optionally, ```"context"``` (top-matched chunks).

## 📊 Evaluation Matrix

The evaluation matrix helps measure the quality, relevance, and reliability of generated answers by combining:

### 1. **Cosine Similarity Scores**

- Measures the semantic closeness between the user query embedding and the embeddings of retrieved chunks.
- Calculated using cosine similarity between query and chunk embeddings.
- Returns:
  - **Average similarity** — mean semantic similarity of top retrieved chunks.
  - **Individual scores** — similarity for each retrieved chunk.
- A higher average indicates better semantic relevance of retrieved context to the query.
<p align="center">
  <img src="images/cosn.png" width="500"/>
</p>

### 2. **Groundedness Check**

- Uses a GPT-4o prompt to verify if the generated answer is strictly supported by the retrieved context.
- Response is either **YES** or **NO**, with a brief explanation.
- Ensures the answer is not hallucinated or unsupported by the knowledge base.

### 3. **Answer Generation**

- Produces a short, precise answer based solely on the retrieved context.
- Designed to avoid long explanations or irrelevant information.
- Focuses on returning the exact name, number, phrase, or fragment that directly answers the question.

## 🧩 Dependencies

| Package           | Version    | Description                               |
|-------------------|------------|-------------------------------------------|
| Python            | 3.8+       | Core programming language                 |
| openai            | ≥ 1.3.7     | OpenAI API for embedding & completion     |
| faiss-cpu         | latest     | Vector similarity search engine (CPU)     |
| tiktoken          | latest     | Tokenizer for OpenAI models               |
| pytesseract       | latest     | OCR engine for Bangla and English text    |
| PyMuPDF           | latest     | PDF text and image extraction             |
| pillow            | latest     | Image processing                          |
| fastapi           | latest     | High-performance web framework            |
| uvicorn[standard] | latest     | ASGI server to run FastAPI                |
| scikit-learn      | latest     | Evaluation and machine learning tools     |
| nest_asyncio      | latest     | Enables nested async loops                |
| streamlit         | latest     | Interactive web-based UI                  |
| python-dotenv     | latest     | Load environment variables from `.env`    |
| numpy             | latest     | Numerical computing library               |

> 📦 To install all dependencies, run:
```bash
pip install -r requirements.txt
```

## 🧾 Questions & Answers

📝 **What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?**

I used PyMuPDF (```fitz```) to render each page of the PDF as an image and Tesseract OCR (via ```pytesseract```) to extract text from those images. This approach was necessary because the PDF contains scanned Bangla text (not selectable or searchable), which made traditional text extraction tools like ```PyPDF2``` or ```pdfminer``` ineffective.

To support Bangla language OCR, I configured Tesseract with the ```ben``` language model and ensured proper setup with ```TESSDATA_PREFIX``` on Windows.

Yes, there were formatting challenges:
- Many pages had inconsistent spacing, broken characters, and misaligned text blocks.
- OCR sometimes misinterpreted compound Bangla characters or punctuation.
- To address this, I used Unicode normalization (```unicodedata```) and regular expressions to clean and standardize the output.

This OCR-based method allowed me to extract readable and indexable Bangla text from the scanned PDF for downstream RAG tasks.

📝 **What chunking strategy did you choose (e.g. paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?**

I used a sentence-based chunking strategy with a 500-token limit per chunk.

- This ensures each chunk is semantically complete and doesn't cut off important context.
- It works well for semantic retrieval because meaningful, token-aware chunks lead to better embedding quality and more accurate search results in FAISS.
- It’s also language-friendly, adapting well to both Bangla and English sentence structures.

📝 **What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?**

I used OpenAI's ```text-embedding-3-large``` model.

- Why? It offers high-quality multilingual support, making it ideal for both Bangla and English queries.
- How? It converts text into dense vectors that capture semantic meaning, allowing similar ideas to be close in vector space—even across languages.
- Result: More accurate chunk retrieval and better answers in our RAG system.

📝 **How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?**

I compare the query with stored chunks using FAISS and L2 (Euclidean) distance on the embedding vectors.

- Why FAISS? It's fast, scalable, and optimized for high-dimensional vector search.
- Why L2 distance? It works well with OpenAI embeddings, providing reliable similarity scoring.
- Storage Setup: We use FAISS for the index and store the corresponding text chunks in a pickle file for quick retrieval.

This setup ensures efficient, accurate semantic search in both Bangla and English.

📝 **How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?**

Ensuring Meaningful Comparison:
- I use OpenAI’s ```text-embedding-3-large``` to embed both the query and the document chunks in the same semantic space, ensuring meaningful comparisons.
- FAISS then retrieves chunks closest in meaning, not just based on keywords.

If the query is vague or lacks context:
- The system may retrieve less relevant chunks, leading to generic or incomplete answers.
- However, the embedding model still tries to infer intent from available cues, often retrieving semantically related content.

To improve performance, we recommend clear and specific queries.

📝 **Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?**

Yes, results are generally relevant thanks to:
- Sentence-based chunking
- High-quality embeddings (```text-embedding-3-large```)
- Semantic search via FAISS

When Results May Seem Irrelevant:
- If chunks are too short or too long → context may be lost or diluted
- If query is vague or ambiguous → embeddings may not capture intent
- If the document is too small → not enough info to match

How to Improve:
- Use dynamic chunking based on meaning and topic
- Fine-tune chunk length (e.g., 300–600 tokens)
- Add metadata filtering or reranking
- Use long-context models like GPT-4o with better prompt design

