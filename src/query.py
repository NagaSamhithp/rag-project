import os
from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

EMBED_MODEL = SentenceTransformer("all-mpnet-base-v2")
CHROMA_CLIENT = chromadb.PersistentClient(path="chroma_db")
COLLECTION = CHROMA_CLIENT.get_collection(name="documents")
GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

print(f"Ready. ChromaDB has {COLLECTION.count()} chunks loaded.")
print()

def retrieve(question: str, top_k: int = 8) -> list[dict]:
    question_vector = EMBED_MODEL.encode([question])[0]
    
    results = COLLECTION.query(
        query_embeddings= [question_vector.tolist()],
        n_results= top_k,
        include= ["documents", "metadatas", "distances"],
    )
    
    return [
        {"text": t, "page": m.get("page","?"), "score": round(d,4)}
        for t, m, d in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

def build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = [
        f"[Source {i} — Page {chunk['page']}]\n{chunk['text']}"
        for i, chunk in enumerate(chunks, start=1)
    ]
    context = "\n\n".join(context_parts)
    return f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
Do not use outside knowledge. If the answer is not in the context, say "I don't find this in the document."

    CONTEXT: {context}

    QUESTION: {question}

    Answer:"""
    
def generate(prompt: str) -> str:
    
    response = GROQ_CLIENT.chat.completions.create(
        
        model= "llama-3.3-70b-versatile",
        temperature= 0.1,
        max_tokens= 1024,
        messages= [
            {
                "role" : "user",
                "content": prompt,
            }
        ],
    )
    
    return response.choices[0].message.content.strip()

def answer(question: str, verbose: bool = False) -> dict:
    
    if verbose:
        print(f" 1. Retrieving chunks for: '{question}'")
    
    chunks = retrieve(question)
    
    if verbose:
        print(f"  2. Retrieved {len(chunks)} chunks (top page: {chunks[0]['page']})")
        print(f"  3. Building prompt")

    prompt = build_prompt(question, chunks)

    if verbose:
        print(f"  4. Sending to Groq (prompt: {len(prompt)} chars)")

    llm_answer = generate(prompt)

    return {
        "question": question,
        "answer":   llm_answer,
        "sources":  chunks,
    }

if __name__ == "__main__":
    test_questions = [
        "What is the attention mechanism?",
        "What are the main contributions of this paper?",
        "How does multi-head attention work?",
        "What datasets were used for experiments?",
        "What is the difference between encoder and decoder?",
    ]

    print("Testing rag with 10 questions")
    print()

    for i, q in enumerate(test_questions, start=1):
        print(f"Question {i}: {q}")
        result = answer(q, verbose=True)

        print()
        print("  ANSWER:")
        print("  " + result["answer"].replace("\n", "\n  "))
        
        print("  SOURCES USED:")
        for src in result["sources"][:2]: 
            print(f"    Page {src['page']} (score: {src['score']}): {src['text'][:100]}...")
        print()