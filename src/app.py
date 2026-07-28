import os
import time
import uuid
import shutil
import tempfile
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

EMBED_MODEL_NAME = "all-mpnet-base-v2"
CHUNK_SIZE       = 700
CHUNK_OVERLAP    = 70        
DEFAULT_TOP_K    = 8
DEFAULT_TEMPERATURE = 0.1
LLM_MODEL        = "llama-3.3-70b-versatile"
UPLOAD_DIR       = "chroma_uploads"    

YOUR_NAME     = "NagaSamhith Patibandla"     
YOUR_GITHUB   = "https://github.com/NagaSamhithp/rag-project"     
YOUR_LINKEDIN = "https://www.linkedin.com/in/naga-samhith-p/"

st.set_page_config(
    page_title = "Document Q&A — RAG System",
    page_icon  = "📄",
    layout     = "wide",
)

@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return OpenAI(
        api_key  = api_key,
        base_url = "https://api.groq.com/openai/v1",
    )


def ingest_pdf(pdf_path, collection_name):
    
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size      = CHUNK_SIZE,
        chunk_overlap   = CHUNK_OVERLAP,
        length_function = len,
    )
    chunks = splitter.split_documents(pages)
    
    if not chunks:
        raise ValueError(
            "No text could be extracted from this PDF."
            "It may be a scanned image rather than a text-based document."
        )
    
    model      = load_embed_model()
    texts      = [c.page_content for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=UPLOAD_DIR)
    
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    
    collection = client.create_collection(name=collection_name)
    collection.add(
        documents  = texts,
        embeddings = embeddings.tolist(),
        metadatas  = [{"page": str(c.metadata.get("page", "?"))} for c in chunks],
        ids        = [f"chunk_{i}" for i in range(len(chunks))],
    )
    return len(pages), len(chunks)

def get_collection(collection_name):
    """Opens the collection for the current document. None if missing."""
    try:
        client = chromadb.PersistentClient(path=UPLOAD_DIR)
        return client.get_collection(name=collection_name)
    except Exception:
        return None

def retrieve(question, collection, top_k):
    """Embed question -> find top_k closest chunks."""
    model = load_embed_model()
    vec   = model.encode([question])[0]

    results = collection.query(
        query_embeddings = [vec.tolist()],
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"],
    )

    return [
        {"text": doc, "page": meta.get("page", "?"), "score": round(dist, 4)}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

def build_prompt(question, chunks):
    """Instruction + numbered context + question."""
    context = "\n\n".join(
        f"[Source {i+1} — Page {c['page']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return f"""You are a helpful assistant. Answer the question using ONLY the context below.
Do not use outside knowledge. If the answer is not in the context, say "I don't find this in the document."

CONTEXT: {context}
QUESTION: {question}

Answer:"""

def generate(prompt, client, temperature):
    """Send prompt to Groq, return the answer text."""
    response = client.chat.completions.create(
        model       = LLM_MODEL,
        temperature = temperature,
        max_tokens  = 1024,
        messages    = [{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


if "history" not in st.session_state:
    st.session_state.history = []

if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False

if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

if "collection_name" not in st.session_state:
    st.session_state.collection_name = f"doc_{uuid.uuid4().hex[:8]}"

if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None


with st.sidebar:
    st.title("📄 Document Q&A")
    st.caption("Ask questions about any PDF")

    st.divider()

    st.subheader("📤 Upload a PDF")
    uploaded = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Any text-based PDF. Scanned images won't work without OCR.",
    )

    if uploaded is not None:
        if st.button("🔄 Process document", use_container_width=True, type="primary"):
            try:
                with st.spinner("Reading, chunking, and embedding..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp_path = tmp.name

                    n_pages, n_chunks = ingest_pdf(
                        tmp_path, st.session_state.collection_name
                    )
                    os.unlink(tmp_path)

                    st.session_state.doc_loaded = True
                    st.session_state.doc_name   = uploaded.name
                    st.session_state.doc_stats  = {"pages": n_pages, "chunks": n_chunks}
                    st.session_state.history    = []

                st.success(f"✅ {n_pages} pages → {n_chunks} chunks")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Could not process this PDF.\n\n{str(e)}")

    if st.session_state.doc_loaded:
        st.divider()
        st.subheader("📊 Current document")
        st.markdown(f"**{st.session_state.doc_name}**")
        stats = st.session_state.doc_stats
        c1, c2 = st.columns(2)
        c1.metric("Pages", stats["pages"])
        c2.metric("Chunks", stats["chunks"])

        if st.button("🗑️ Remove document", use_container_width=True):
            try:
                client = chromadb.PersistentClient(path=UPLOAD_DIR)
                client.delete_collection(st.session_state.collection_name)
            except Exception:
                pass
            st.session_state.doc_loaded = False
            st.session_state.doc_name   = None
            st.session_state.doc_stats  = None
            st.session_state.history    = []
            st.rerun()

        st.divider()

        with st.expander("⚙️ Advanced settings"):
            st.caption(
                "These are already tuned to the best-performing configuration "
                "from testing. Only change them if you want to experiment."
            )

            top_k = st.slider(
                "Passages to retrieve",
                min_value = 1,
                max_value = 12,
                value     = DEFAULT_TOP_K,
                help = "How many pieces of the document to search through per question. "
                       "More gives broader context but can add irrelevant text.",
            )

            temperature = st.slider(
                "Creativity (temperature)",
                min_value = 0.0,
                max_value = 1.0,
                value     = DEFAULT_TEMPERATURE,
                step      = 0.1,
                help = "Low values keep answers factual and grounded in the document. "
                       "High values make the model more creative, which risks inventing "
                       "details. Recommended: keep at 0.1 for document Q&A.",
            )

            if temperature > 0.4:
                st.warning(
                    "⚠️ High creativity increases the risk of the model "
                    "inventing details not in your document.",
                    icon="⚠️",
                )
    else:
        top_k       = DEFAULT_TOP_K
        temperature = DEFAULT_TEMPERATURE

    st.divider()

    st.subheader("🔧 Tech Stack")
    st.markdown(
        f"""
        - **LLM:** `{LLM_MODEL}` via Groq
        - **Embeddings:** `{EMBED_MODEL_NAME}`
        - **Vector DB:** ChromaDB
        - **Chunking:** {CHUNK_SIZE} chars / {CHUNK_OVERLAP} overlap
        - **Evaluation:** RAGAS
        - **UI:** Streamlit
        """
    )

    st.caption("Config tuned via ablation study — see repo for RAGAS scores.")

    st.divider()

    st.subheader("👤 Built by")
    st.markdown(f"**{YOUR_NAME}**")
    st.markdown(f"[GitHub]({YOUR_GITHUB}) · [LinkedIn]({YOUR_LINKEDIN})")

groq_client = load_groq_client()


if groq_client is None:
    st.error(
        "❌ **GROQ_API_KEY not found.** Add it to your `.env` file:\n\n"
        "```\nGROQ_API_KEY=gsk_your_key_here\n```"
    )
    st.stop()


if not st.session_state.doc_loaded:
    st.title("Ask questions about any PDF")
    st.markdown(
        "Upload a document in the sidebar and ask anything about it. "
        "Every answer shows the exact source passages it came from, "
        "so you can verify nothing is made up."
    )

    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 1️⃣ Upload")
        st.caption("Drop any text-based PDF into the sidebar.")
    with c2:
        st.markdown("### 2️⃣ Ask")
        st.caption("Type any question. The system finds the most relevant passages automatically.")
    with c3:
        st.markdown("### 3️⃣ Verify")
        st.caption("Expand the sources under each answer to see exactly which pages it used.")

    st.divider()
    st.info("👈 Start by uploading a PDF in the sidebar", icon="📤")
    st.stop()


collection = get_collection(st.session_state.collection_name)


if collection is None:
    st.error("Document database was lost. Please re-upload your PDF.")
    st.session_state.doc_loaded = False
    st.stop()


st.title("Ask questions about your document")
st.caption(f"📄 **{st.session_state.doc_name}** · {collection.count()} searchable chunks")


question = st.text_input(
    "Your question",
    placeholder="e.g. What is the main contribution of this paper?",
    label_visibility="collapsed",
)


col1, col2 = st.columns([1, 5])
with col1:
    ask = st.button("🔍 Ask", type="primary", use_container_width=True)
with col2:
    if st.session_state.history:
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()


if ask and question.strip():
    try:
        with st.spinner("Searching document and generating answer..."):
            t0 = time.time()
            chunks  = retrieve(question, collection, top_k)
            prompt  = build_prompt(question, chunks)
            answer  = generate(prompt, groq_client, temperature)
            elapsed = round(time.time() - t0, 2)

        st.session_state.history.insert(0, {
            "question": question,
            "answer":   answer,
            "chunks":   chunks,
            "seconds":  elapsed,
        })
    except Exception as e:
        st.error(f"❌ Something went wrong.\n\n{str(e)}")

elif ask and not question.strip():
    st.warning("Please type a question first.")


for item in st.session_state.history:
    st.divider()
    st.markdown(f"### ❓ {item['question']}")

    st.markdown(
        f"""
        <div style="
            background-color: #1a2332;
            border-left: 4px solid #10b981;
            border-radius: 6px;
            padding: 16px 20px;
            margin: 12px 0;
            line-height: 1.7;
        ">{item['answer']}</div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(f"⏱️ {item['seconds']}s · {len(item['chunks'])} sources used")

    with st.expander(f"📖 View the {len(item['chunks'])} source passages"):
        for j, chunk in enumerate(item["chunks"], 1):
            st.markdown(
                f"**Source {j}** — Page {chunk['page']} · "
                f"distance `{chunk['score']}` *(lower = more similar)*"
            )
            st.text(chunk["text"])
            if j < len(item["chunks"]):
                st.markdown("---")


if not st.session_state.history:
    st.markdown("")
    st.markdown("**Try asking:**")
    for ex in [
        "What is the main contribution of this document?",
        "Summarize the methodology.",
        "What are the key findings?",
    ]:
        st.markdown(f"- {ex}")