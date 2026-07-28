import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

PDF_Path = "data/paper.pdf"
CHROMA_DIR = "chroma_db"
COLLECTION = "documents"

loader = PyPDFLoader(PDF_Path)
pages = loader.load()

print(f"Loaded {len(pages)} pages")

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap = 70,
    length_function = len,
)

chunks = splitter.split_documents(pages)

model = SentenceTransformer("all-mpnet-base-v2")
texts = [c.page_content for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True)

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION)

if collection.count() > 0:
    print(f"  Clearing {collection.count()} old chunks...")
    collection.delete(where={"source": {"$ne": ""}})
    
ids = [f"chunk_{i}" for i in range(len(chunks))]
metadatas = [
    {
        "page": str(c.metadata.get("page", "?")),
        "source": c.metadata.get("source", PDF_Path),
    }
    for c in chunks
]

collection.add(
    documents= texts,
    embeddings= embeddings.tolist(),
    metadatas= metadatas,
    ids = ids,
)

print(f"Stored {collection.count()} chunks in ChromaDB")

TEST_QUESTION = "What is attention mechanism?"

question_embedding = model.encode([TEST_QUESTION])[0]

results = collection.query(
    query_embeddings= [question_embedding.tolist()],
    n_results= 3,
    include= ["documents", "metadatas", "distances"]
)

print(f"Query: '{TEST_QUESTION}'")
print()

for i, (doc, meta, dist) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
)):
    print(f"Result {i+1}  |  page: {meta['page']}  |  distance: {dist:.4f}")
    print(doc[:300])
    print()
