import os
import json
import csv
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import llm_factory

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)

from langchain_huggingface import HuggingFaceEmbeddings as LCEmbeddings


PDF_PATH        = "data/paper.pdf"
COLLECTION_NAME = "documents"
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")


JUDGE_MODEL = "llama-3.1-8b-instant"

judge_llm = llm_factory(
    model    = JUDGE_MODEL,
    provider = "openai",
    client   = OpenAI(
        api_key  = GROQ_API_KEY,
        base_url = "https://api.groq.com/openai/v1",
    ),
)

judge_embeddings = LCEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

faithfulness.llm            = judge_llm
context_precision.llm       = judge_llm
answer_relevancy.llm        = judge_llm
answer_relevancy.embeddings = judge_embeddings

groq_gen_client = OpenAI(
    api_key  = GROQ_API_KEY,
    base_url = "https://api.groq.com/openai/v1",
)


TEST_DATA = [
    {
        "question":  "What is the Transformer model?",
        "reference": "The Transformer is a model architecture that relies entirely on an attention mechanism to draw global dependencies between input and output, dispensing with recurrence and convolutions entirely.",
    },
    {
        "question":  "What optimizer was used to train the Transformer?",
        "reference": "The Adam optimizer was used with beta1=0.9, beta2=0.98, and epsilon=10 to the power of negative 9.",
    },
    {
        "question":  "How many attention heads does the base Transformer model use?",
        "reference": "The base model uses 8 parallel attention heads.",
    },
    {
        "question":  "What is the model dimension dmodel of the base Transformer?",
        "reference": "The model dimension dmodel is 512 for the base model.",
    },
    {
        "question":  "What datasets were used to train the Transformer?",
        "reference": "The WMT 2014 English-German dataset with about 4.5 million sentence pairs and the WMT 2014 English-French dataset with 36 million sentences were used.",
    },
    {
        "question":  "What BLEU score did the big Transformer achieve on English-German translation?",
        "reference": "The big Transformer model achieved 28.4 BLEU on the WMT 2014 English-to-German translation task, outperforming all previously published models.",
    },
    {
        "question":  "What is the purpose of positional encoding in the Transformer?",
        "reference": "Positional encodings are added to the input embeddings to inject information about the relative or absolute position of tokens in the sequence, since the model contains no recurrence or convolution.",
    },
    {
        "question":  "What is multi-head attention?",
        "reference": "Multi-head attention linearly projects queries, keys, and values h times with different learned projections, performs attention in parallel on each projection, then concatenates and projects the results.",
    },
    {
        "question":  "What is the inner-layer dimensionality of the feed-forward networks?",
        "reference": "The inner-layer dimensionality of the position-wise feed-forward networks is 2048.",
    },
    {
        "question":  "Why did the authors use scaled dot-product attention instead of additive attention?",
        "reference": "Scaled dot-product attention is faster and more space-efficient in practice because it can be implemented using optimized matrix multiplication code, while additive attention uses a feed-forward network and is slower.",
    },
]


def ingest_with_config(chunk_size, embed_model_name, chroma_dir):
    print(f"    Ingesting (chunk_size={chunk_size}, model={embed_model_name.split('/')[-1]})")

    loader = PyPDFLoader(PDF_PATH)
    pages  = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = chunk_size,
        chunk_overlap = max(20, chunk_size // 10),
    )
    chunks = splitter.split_documents(pages)

    model      = SentenceTransformer(embed_model_name)
    texts      = [c.page_content for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)

    os.makedirs(chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)
    collection.add(
        documents  = texts,
        embeddings = embeddings.tolist(),
        metadatas  = [{"page": str(c.metadata.get("page", "?"))} for c in chunks],
        ids        = [f"chunk_{i}" for i in range(len(chunks))],
    )
    print(f"    Stored {collection.count()} chunks")
    return len(chunks)


def evaluate_config(chroma_dir, embed_model_name, top_k, generation_model):
    print(f"    Evaluating (top_k={top_k}, gen={generation_model})")

    embed_model = SentenceTransformer(embed_model_name)
    client      = chromadb.PersistentClient(path=chroma_dir)
    collection  = client.get_collection(name=COLLECTION_NAME)

    def retrieve(question):
        vec = embed_model.encode([question])[0]
        res = collection.query(
            query_embeddings = [vec.tolist()],
            n_results        = top_k,
            include          = ["documents"],
        )
        return res["documents"][0]

    def build_prompt(question, contexts):
        ctx = "\n\n".join(f"[Source {i+1}]\n{c}" for i, c in enumerate(contexts))
        return f"""Answer using ONLY the context below. Do not use outside knowledge.
If the answer is not in the context, say "I don't find this in the document."

        CONTEXT: {ctx}
        QUESTION: {question}
        Answer:"""

    def generate(prompt):
        resp = groq_gen_client.chat.completions.create(
            model       = generation_model,
            temperature = 0.1,
            max_tokens  = 512,
            messages    = [{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    rows = []
    for item in TEST_DATA:
        q        = item["question"]
        contexts = retrieve(q)
        prompt   = build_prompt(q, contexts)
        answer   = generate(prompt)
        rows.append({
            "question":      q,
            "answer":        answer,
            "contexts":      contexts,
            "ground_truths": item["reference"],
            "reference":     item["reference"],
        })

    ds = Dataset.from_list(rows)

    run_config = RunConfig(max_workers=2, timeout=180)

    results = evaluate(
        dataset    = ds,
        metrics    = [faithfulness, answer_relevancy, context_precision],
        run_config = run_config,
    )

    def safe_avg(values):
        clean = [v for v in values if v == v]
        return round(sum(clean) / len(clean), 4) if clean else 0.0

    scores = {
        "faithfulness":      safe_avg(results["faithfulness"]),
        "answer_relevancy":  safe_avg(results["answer_relevancy"]),
        "context_precision": safe_avg(results["context_precision"]),
    }
    scores["average"] = round(sum(scores.values()) / 3, 4)

    print(f"    F={scores['faithfulness']}  AR={scores['answer_relevancy']}  "
          f"CP={scores['context_precision']}  AVG={scores['average']}")
    return scores


if __name__ == "__main__":
    MINILM = "sentence-transformers/all-MiniLM-L6-v2"
    MPNET  = "sentence-transformers/all-mpnet-base-v2"
    GEN_8B  = "llama-3.1-8b-instant"
    GEN_70B = "llama-3.3-70b-versatile"


    experiments = [
        ("E0_baseline",     500,  5, MINILM, GEN_8B,  None),
        ("E1_chunk300",     300,  5, MINILM, GEN_8B,  None),
        ("E2_chunk700",     700,  5, MINILM, GEN_8B,  None),
        ("E3_topk3",        500,  3, MINILM, GEN_8B,  "E0_baseline"),
        ("E4_topk8",        500,  8, MINILM, GEN_8B,  "E0_baseline"),
        ("E5_mpnet",        500,  5, MPNET,  GEN_8B,  None),
        ("E6_best",         700,  8, MINILM, GEN_8B,  None),
        ("E8_topk12",       700, 12, MINILM, GEN_8B,  "E6_best"),
        ("E9_chunk400",     400,  8, MINILM, GEN_8B,  None),
        ("E10_mpnet_best",  700,  8, MPNET,  GEN_8B,  None),
        ("E11_production",  700,  8, MPNET,  GEN_70B, "E10_mpnet_best"),
    ]

    all_results = []
    os.makedirs("tests", exist_ok=True)

    for (name, chunk_size, top_k, embed_model, gen_model, reuse_from) in experiments:
        print(f"\n{'='*66}")
        print(f"EXPERIMENT: {name}")
        print(f"  chunk={chunk_size}  top_k={top_k}  "
              f"embed={embed_model.split('/')[-1]}  gen={gen_model}")
        print('='*66)

        if reuse_from is None:
            chroma_dir = f"chroma_experiments/{name}"
            n_chunks   = ingest_with_config(chunk_size, embed_model, chroma_dir)
        else:
            chroma_dir = f"chroma_experiments/{reuse_from}"
            n_chunks   = f"reused from {reuse_from}"
            print(f"    Reusing {reuse_from}'s ChromaDB (embedding model unchanged)")

        scores = evaluate_config(chroma_dir, embed_model, top_k, gen_model)

        all_results.append({
            "experiment":       name,
            "chunk_size":       chunk_size,
            "top_k":            top_k,
            "embed_model":      embed_model.split('/')[-1],
            "generation_model": gen_model,
            "n_chunks":         n_chunks,
            "scores":           scores,
        })

        with open("tests/optimization_full_progress.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"    Progress saved ({len(all_results)}/{len(experiments)} done)")


    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"tests/optimization_full_{ts}.json"
    csv_path  = f"tests/comparison_full_{ts}.csv"

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Experiment", "chunk_size", "top_k", "embed_model",
                    "generation_model", "Faithfulness", "AnswerRelevancy",
                    "ContextPrecision", "Average"])
        for r in all_results:
            s = r["scores"]
            w.writerow([r["experiment"], r["chunk_size"], r["top_k"],
                        r["embed_model"], r["generation_model"],
                        s["faithfulness"], s["answer_relevancy"],
                        s["context_precision"], s["average"]])


    print(f"\n\n{'='*86}")
    print("FULL COMPARISON TABLE  (E0 -> E11)")
    print('='*86)
    print(f"{'Experiment':<18}{'Chunk':>7}{'TopK':>6}{'Model':>9}{'Gen':>6}"
          f"{'Faith':>8}{'AnsRel':>8}{'CtxPrec':>9}{'AVG':>8}")
    print('-'*86)

    best = max(all_results, key=lambda r: r["scores"]["average"])

    for r in all_results:
        s   = r["scores"]
        mdl = "MPNet" if "mpnet" in r["embed_model"].lower() else "MiniLM"
        gen = "70b" if "70b" in r["generation_model"] else "8b"
        tag = "  <- BEST" if r["experiment"] == best["experiment"] else ""
        print(f"{r['experiment']:<18}{r['chunk_size']:>7}{r['top_k']:>6}{mdl:>9}{gen:>6}"
              f"{s['faithfulness']:>8.4f}{s['answer_relevancy']:>8.4f}"
              f"{s['context_precision']:>9.4f}{s['average']:>8.4f}{tag}")

    print(f"\nFinal production config : {best['experiment']}  (avg={best['scores']['average']})")
    print(f"Improvement over baseline: "
          f"{round((best['scores']['average'] / all_results[0]['scores']['average'] - 1) * 100, 1)}%")
    print(f"JSON saved : {json_path}")
    print(f"CSV saved  : {csv_path}")





















