# 🔍 Groundly — RAG System with Proven, Measured Quality

![Python](https://img.shields.io/badge/Python-3.12-3670A0?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3%2070B-black)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-yellow)
![RAGAS](https://img.shields.io/badge/Evaluated%20with-RAGAS-green)
![LangSmith](https://img.shields.io/badge/Observability-LangSmith-1C3C3C)

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-groundly.streamlit.app-brightgreen?style=for-the-badge)](https://groundly.streamlit.app)

**Ask any PDF a question. Get an answer backed by the exact source passage — not a guess.**

Most RAG demos ship once and call it done. This one didn't stop there: baseline → measured with RAGAS → **11-experiment ablation study** → deployed → **fully traced in production**. Every claim on this page is backed by a number, a screenshot, or a script you can run yourself.

---

## 📊 Results at a glance

![Results at a Glance](screenshots/results_at_a_glance.png)

Scores of 0.73–0.88 sit in the range considered strong for production RAG systems — this isn't a demo that works once, it's a system that was measured, broken, fixed, and measured again. **11 experiments. Full history in [`src/optimize.py`](src/optimize.py) and [`tests/`](tests/).**

---

## ⭐ Why this stands out

- 🎯 **Measured, not assumed** — every configuration change validated with RAGAS across 3 metrics, not shipped on a hunch
- 🔬 **11 tracked experiments** — baseline to production, including the ones that made things *worse* (and why)
- 🩺 **Honest about limitations** — documents exactly where retrieval still fails and why, instead of hiding it
- 📡 **Fully observable** — every query traced with latency, token, and cost data via LangSmith
- 🚀 **Actually deployed** — live URL, not just a repo someone has to clone to see it work

---

## 🎬 What it does

1. **Upload** any text-based PDF
2. **Ask** a question in plain English
3. **Verify** — every answer shows the exact source passage it came from

If the answer isn't in the document, the system says so — it doesn't guess.

---

## 🏗 Architecture

![Architecture Diagram](screenshots/architecture_diagram.png)

**Pipeline:**

1. **Ingestion** — PyPDFLoader extracts text → `RecursiveCharacterTextSplitter` chunks it → `sentence-transformers` embeds each chunk → stored in ChromaDB
2. **Retrieval** — question is embedded with the same model, top-k most similar chunks pulled via cosine similarity
3. **Generation** — retrieved chunks + question assembled into a prompt, sent to Llama 3.3 70B via Groq, with an explicit "answer only from context" instruction
4. **Observability** — every call traced through LangSmith, capturing latency, token usage, and full input/output

---

## 🔧 Tech Stack

![Tech Stack](screenshots/tech_stack_table.png)

---

## 🧪 Evaluation methodology

Quality measured with [RAGAS](https://github.com/explodinggradients/ragas) across three metrics on a fixed 10-question test set built from the source document ("Attention Is All You Need"):

![Evaluation Methodology](screenshots/evaluation_methodology.png)

---

## 🔬 Ablation study — 11 experiments, one variable at a time

![Ablation Results](screenshots/ablation_results_table.png)

**🏆 Result: +68% relative improvement (0.431 → 0.727)**

### Key findings

> **Finding 1 — Embedding quality only pays off in combination.**
> MiniLM → MPNet alone (E5) *underperformed* baseline. It only became a real win (E10) once paired with the right chunk size and retrieval breadth. Testing one variable in isolation understated its true value.

> **Finding 2 — Retrieval and generation are independent levers.**
> E10 → E11 held retrieval completely constant and only upgraded the generation model. Answer Relevancy nearly doubled (0.446 → 0.877). Context Precision stayed flat — exactly as expected, since retrieval logic never changed. This is direct proof the two levers don't substitute for each other.

Full experiment history: [`src/optimize.py`](src/optimize.py) · Raw results: [`tests/`](tests/)

---

## 🩺 Known limitations

The system reliably answers architecture and mechanism questions ("What is multi-head attention?", "What is positional encoding for?"). It consistently **fails** on narrowly-scoped factual questions where the answer is a short sentence buried in a section dense with unrelated detail — e.g. "What datasets were used to train the model?"

This was tested across **5+ configurations** (chunk sizes 300–700, top-k 5–12, both embedding models, both generation models) and the failure persisted identically every time — confirming it's a **retrieval** limitation, not a generation one. The system correctly responds "I don't find this in the document" rather than hallucinating — the intended safe failure mode.

**Not yet implemented, likely fix:** hybrid keyword + semantic search, or section-aware metadata filtering.

---

## 📡 Observability

Every query is traced through LangSmith — the Groq client wrapped with `wrap_openai()`, retrieval instrumented with `@traceable`, capturing latency, token usage, and full input/output per pipeline step.

![Observability Metrics](screenshots/observability_metrics.png)

![LangSmith traces](screenshots/langsmith-traces-overview.png)

---

## ⚙️ Setup

**Requirements:** Python 3.12, a [Groq API key](https://console.groq.com) (free tier), optionally a [LangSmith API key](https://smith.langchain.com) for tracing.

```bash
# Clone and enter the project
git clone https://github.com/NagaSamhithp/rag-project.git
cd rag-project

# Create and activate a virtual environment
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# then edit .env and add your GROQ_API_KEY
```

**Add your document:** place a PDF at `data/paper.pdf`, then run:

```bash
python src/ingest.py
```

**Ask questions from the terminal:**

```bash
python src/query.py
```

**Or launch the web UI:**

```bash
streamlit run src/app.py
```

Then open `http://localhost:8501` and upload any PDF.

---

## 📁 Project structure

```
rag-project/
├── data/                       # source PDFs
├── src/
│   ├── ingest.py                # chunk + embed + store pipeline
│   ├── query.py                 # retrieval + generation pipeline
│   ├── app.py                   # Streamlit web UI (upload-only)
│   ├── optimize.py              # full E0-E11 ablation study
│   └── evaluate_baseline.py     # Day 4 baseline RAGAS run
├── tests/                       # RAGAS results (JSON + CSV)
├── screenshots/                 # architecture, results, and trace images
├── requirements.txt
└── .env.example
```

---

## 👨‍💻 Author

**NagaSamhith Patibandla**

[![GitHub](https://img.shields.io/badge/GitHub-NagaSamhithp-181717?logo=github)](https://github.com/NagaSamhithp)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/naga-samhith-p/)