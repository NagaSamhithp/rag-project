import os
import json
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.query import retrieve, build_prompt, generate

from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)
from langchain_huggingface import HuggingFaceEmbeddings as LCEmbeddings


judge_llm = llm_factory(
    model    = "llama-3.1-8b-instant",
    provider = "openai",
    client   = OpenAI(
        api_key  = os.getenv("GROQ_API_KEY"),
        base_url = "https://api.groq.com/openai/v1",
    ),
)

lc_embeddings = LCEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

faithfulness.llm            = judge_llm
context_precision.llm       = judge_llm
answer_relevancy.llm        = judge_llm
answer_relevancy.embeddings = lc_embeddings


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


print("Running RAG on 10 questions")
print()

rows = []
for i, row in enumerate(TEST_DATA, 1):
    q = row["question"]
    print(f"  [{i:02d}/10] {q[:50]}...")
    chunks = retrieve(q, top_k=5)
    prompt = build_prompt(q, chunks)
    answer = generate(prompt)
    rows.append({
        "question":      q,
        "answer":        answer,
        "contexts":      [c["text"] for c in chunks],
        "ground_truths": row["reference"], 
        "reference": row["reference"],
    })

print(f"\n  Done. {len(rows)} rows collected.\n")


ds = Dataset.from_list(rows)
print(f"Shape  : {ds.shape}")
print(f"Columns: {ds.column_names}\n")

print("Running RAGAS")
print()

results = evaluate(
    dataset = ds,
    metrics = [faithfulness, answer_relevancy, context_precision],
)


def safe_avg(values):
    clean = [v for v in values if v == v]  
    return round(sum(clean) / len(clean), 4) if clean else 0.0

def bar(s):
    s = 0.0 if s != s else s
    return "█" * int(s*20) + "░" * (20-int(s*20))

def lbl(s):
    if s != s: return "NaN"
    return " Good" if s>=0.8 else ("  Needs work" if s>=0.6 else " Poor")

f_score  = safe_avg(results["faithfulness"])
ar_score = safe_avg(results["answer_relevancy"])
cp_score = safe_avg(results["context_precision"])
avg      = round((f_score + ar_score + cp_score) / 3, 4)

print("RESULTS")

print(f"\n  {'Faithfulness':<24} {f_score:.4f}  [{bar(f_score)}]  {lbl(f_score)}")
print(f"  {'Answer Relevancy':<24} {ar_score:.4f}  [{bar(ar_score)}]  {lbl(ar_score)}")
print(f"  {'Context Precision':<24} {cp_score:.4f}  [{bar(cp_score)}]  {lbl(cp_score)}")
print(f"\n  {'AVERAGE':<24} {avg:.4f}\n")


os.makedirs("tests", exist_ok=True)
ts  = datetime.now().strftime("%Y%m%d_%H%M%S")

out = {
    "timestamp":     datetime.now().isoformat(),
    "version":       "baseline_v1",
    "ragas_version": "0.4.3",
    "config": {
        "chunk_size":    500,
        "chunk_overlap": 50,
        "top_k":         5,
        "judge_llm":     "llama-3.1-8b-instant",
        "embed_model":   "all-MiniLM-L6-v2",
    },
    "scores": {
        "faithfulness":      f_score,
        "answer_relevancy":  ar_score,
        "context_precision": cp_score,
        "average":           avg,
    },
    "num_questions": len(TEST_DATA),
}
path = f"tests/eval_{ts}.json"
with open(path, "w") as f:
    json.dump(out, f, indent=2)

print(f"  Saved: {path}")