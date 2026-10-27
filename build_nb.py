import nbformat as nbf
nb = nbf.v4.new_notebook()
C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s))
code=lambda s:C.append(nbf.v4.new_code_cell(s))

md("""# RAG Pipeline: PDF Chunking and Retrieval
**Stack:** Python · LangChain · FAISS · Groq (LLM) · HuggingFace sentence-transformers (embeddings)

**Pipeline:** PDFs → load → chunk → embed → FAISS index → semantic search → (Groq) grounded answer

> **Why not OpenAI?** The project brief mentions OpenAI embeddings, but no OpenAI key is available. Groq serves fast LLM inference
> but **does not provide an embeddings endpoint**, so embeddings are generated locally with `all-MiniLM-L6-v2` (free, no key) and Groq is used for the answer-generation step.""")

md("## 1. Setup")
code("""# Run once (remove the leading % if running as a plain script)
%pip install -q langchain langchain-community langchain-text-splitters langchain-groq langchain-huggingface sentence-transformers faiss-cpu pypdf python-dotenv scikit-learn pandas""")
code("""import os, json, glob, time
from pathlib import Path
import numpy as np, pandas as pd
from dotenv import load_dotenv

load_dotenv()                                  # reads GROQ_API_KEY from a .env file
DATA_DIR, OUT_DIR = Path("data"), Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

GROQ_MODEL   = "llama-3.3-70b-versatile"       # any current Groq chat model works
CHUNK_SIZE, CHUNK_OVERLAP = 400, 80
TOP_K = 3

# Set RAG_OFFLINE=1 if the HuggingFace model cannot be downloaded (e.g. restricted network).
OFFLINE = os.getenv("RAG_OFFLINE") == "1"
print("Groq key found:", bool(os.getenv("GROQ_API_KEY")), "| offline embeddings:", OFFLINE)""")

md("## 2. Load the PDFs\n`PyPDFLoader` returns one `Document` per page and keeps `source` and `page` in the metadata, which we use later for citations.")
code("""from langchain_community.document_loaders import PyPDFLoader

pdf_files = sorted(DATA_DIR.glob("*.pdf"))
docs = []
for f in pdf_files:
    pages = PyPDFLoader(str(f)).load()
    for p in pages:
        p.metadata["source"] = f.name
    docs.extend(pages)
    print(f"{f.name:35s} {len(pages)} pages, {sum(len(p.page_content) for p in pages):>6} characters")
print("Total pages loaded:", len(docs))""")

md("## 3. Chunking\n`RecursiveCharacterTextSplitter` tries to split on paragraph, then line, then sentence, then word boundaries, so chunks stay semantically coherent. The overlap keeps context that straddles a boundary.")
code("""from langchain_text_splitters import RecursiveCharacterTextSplitter

def make_chunks(documents, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
                                              separators=["\\n\\n", "\\n", ". ", " ", ""])
    chunks = splitter.split_documents(documents)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks

chunks = make_chunks(docs)
lens = [len(c.page_content) for c in chunks]
print(f"{len(chunks)} chunks | avg {np.mean(lens):.0f} chars | min {min(lens)} | max {max(lens)}")
print("\\nExample chunk:\\n", chunks[3].page_content, "\\n", chunks[3].metadata)""")

md("## 4. Embeddings\nPrimary: `sentence-transformers/all-MiniLM-L6-v2` (384-dim). Fallback: a TF-IDF + SVD (LSA) embedder, used only when `RAG_OFFLINE=1`.")
code("""from langchain_core.embeddings import Embeddings

class TfidfSvdEmbeddings(Embeddings):
    \"\"\"Offline fallback so the pipeline still runs without downloading a model.\"\"\"
    def __init__(self, corpus, dim=64):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        X = self.vec.fit_transform(corpus)
        self.svd = TruncatedSVD(n_components=min(dim, X.shape[0] - 1, X.shape[1] - 1), random_state=0)
        self.svd.fit(X)
    def _enc(self, texts):
        Z = self.svd.transform(self.vec.transform(texts))
        return (Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)).astype("float32").tolist()
    def embed_documents(self, texts): return self._enc(texts)
    def embed_query(self, text):      return self._enc([text])[0]

def get_embeddings(chunk_list):
    if OFFLINE:
        return TfidfSvdEmbeddings([c.page_content for c in chunk_list])
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                 encode_kwargs={"normalize_embeddings": True})

embeddings = get_embeddings(chunks)
print("Embedding backend:", type(embeddings).__name__, "| vector dim:", len(embeddings.embed_query("test")))""")

md("## 5. Build and persist the FAISS vector store")
code("""from langchain_community.vectorstores import FAISS

t0 = time.time()
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
print(f"Indexed {vectorstore.index.ntotal} vectors in {time.time()-t0:.1f}s -> saved to ./faiss_index")""")

md("## 6. Semantic retrieval\nFAISS returns an **L2 distance** (lower = more similar). Each result is shown with its source file and page.")
code("""def retrieve(query, k=TOP_K, store=None):
    store = store or vectorstore
    return store.similarity_search_with_score(query, k=k)

def show(query, k=TOP_K):
    print("=" * 90, "\\nQUESTION:", query)
    for rank, (doc, score) in enumerate(retrieve(query, k), 1):
        print(f"\\n  #{rank}  distance={score:.3f}  [{doc.metadata['source']} | page {doc.metadata['page']+1}]")
        print("     ", doc.page_content.replace("\\n", " ")[:300])

show("What documents do I need to bring for admission?")""")

md("## 7. Sample queries and outputs\nEach query has an expected source file so we can measure retrieval quality (hit@1 / hit@3).")
code("""test_queries = [
    ("What documents do I need to bring for admission?",               "patient_admission_discharge_guide.pdf", "photo ID"),
    ("What are the visiting hours for the general ward?",              "patient_admission_discharge_guide.pdf", "5 PM to 7 PM"),
    ("By what time must I leave the room on the day of discharge?",    "patient_admission_discharge_guide.pdf", "before 11 AM"),
    ("How long does cashless pre-authorisation take for a planned admission?", "insurance_billing_faq.pdf", "4 hours"),
    ("Which documents are provided for an insurance reimbursement claim?",     "insurance_billing_faq.pdf", "itemised bill"),
    ("How long do I have to raise a billing dispute?",                 "insurance_billing_faq.pdf", "30 days"),
    ("How early should I arrive for my outpatient appointment?",       "outpatient_medication_safety_handbook.pdf", "15 minutes"),
    ("How long should I fast before a fasting blood sugar test?",      "outpatient_medication_safety_handbook.pdf", "8 to 10 hours"),
    ("What should I do if a medicine gives me a side effect?",         "outpatient_medication_safety_handbook.pdf", "pharmacovigilance"),
]
for q, _, _ in test_queries[:3]:
    show(q)""")

code("""norm = lambda t: " ".join(t.lower().split())   # collapse PDF line-wraps before keyword matching

def evaluate(store, queries=test_queries, k=TOP_K):
    rows = []
    for q, exp_src, keyword in queries:
        res = retrieve(q, k, store)
        srcs = [d.metadata["source"] for d, _ in res]
        rows.append({"question": q, "expected_source": exp_src,
                     "top1_source": srcs[0],
                     "hit@1 (source)": srcs[0] == exp_src,
                     f"hit@{k} (source)": exp_src in srcs,
                     "answer_in_top1": norm(keyword) in norm(res[0][0].page_content),
                     f"answer_in_top{k}": any(norm(keyword) in norm(d.page_content) for d, _ in res),
                     "top1_distance": round(float(res[0][1]), 3)})
    return pd.DataFrame(rows)

results = evaluate(vectorstore)
display(results)
print("\\nsource hit@1:", results['hit@1 (source)'].mean().round(2),
      "| answer-in-top1:", results['answer_in_top1'].mean().round(2),
      f"| answer-in-top{TOP_K}:", results[f'answer_in_top{TOP_K}'].mean().round(2))
results.to_csv(OUT_DIR / "retrieval_evaluation.csv", index=False)""")

code("""# Save the full retrieval transcript for the submission
lines = ["# Sample retrieval outputs\\n"]
for q, _, _ in test_queries:
    lines.append(f"## Q: {q}\\n")
    for rank, (d, s) in enumerate(retrieve(q), 1):
        lines.append(f"**#{rank}** · distance `{s:.3f}` · `{d.metadata['source']}` p.{d.metadata['page']+1}\\n")
        lines.append("> " + d.page_content.replace("\\n", " ") + "\\n")
(OUT_DIR / "sample_outputs.md").write_text("\\n".join(lines), encoding="utf-8")
print("Saved outputs/sample_outputs.md")""")

md("## 8. Experiment: effect of chunk size\nSame queries, three chunk sizes. This feeds the analysis report.")
code("""rows = []
for size in (200, 400, 800):
    ch = make_chunks(docs, size=size, overlap=size // 5)
    emb = get_embeddings(ch)
    store = FAISS.from_documents(ch, emb)
    ev = evaluate(store)
    rows.append({"chunk_size": size, "overlap": size // 5, "num_chunks": len(ch),
                 "source_hit@1": ev["hit@1 (source)"].mean(),
                 "answer_in_top1": ev["answer_in_top1"].mean(),
                 f"answer_in_top{TOP_K}": ev[f"answer_in_top{TOP_K}"].mean()})
exp = pd.DataFrame(rows).round(2)
display(exp)
exp.to_csv(OUT_DIR / "chunk_size_experiment.csv", index=False)""")

md("## 9. Grounded answers with Groq\nThe retrieved chunks are passed to a Groq-hosted LLM, which is instructed to answer **only** from the context. Requires `GROQ_API_KEY` in a `.env` file (free key at console.groq.com).")
code("""from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a hospital patient-information assistant. Answer ONLY from the provided context. "
               "If the answer is not in the context, say you could not find it. Cite sources as [file, page]. "
               "Do not give medical advice or diagnoses; for clinical questions, tell the patient to contact their doctor."),
    ("human", "Context:\\n{context}\\n\\nQuestion: {question}"),
])

def answer(question, k=TOP_K):
    from langchain_groq import ChatGroq
    hits = retrieve(question, k)
    context = "\\n\\n".join(f"[{d.metadata['source']}, page {d.metadata['page']+1}]\\n{d.page_content}" for d, _ in hits)
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    return (prompt | llm).invoke({"context": context, "question": question}).content

if os.getenv("GROQ_API_KEY"):
    qa = []
    for q, _, _ in test_queries:
        a = answer(q); qa.append({"question": q, "answer": a})
        print("Q:", q, "\\nA:", a, "\\n")
    (OUT_DIR / "groq_answers.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
else:
    print("GROQ_API_KEY not set - skipping generation. Add it to .env and re-run this cell.")""")

md("""## 10. Conclusion
The pipeline loads three PDFs, splits them into overlapping chunks, embeds them, stores them in FAISS and retrieves the most relevant chunks for a question; Groq then writes a grounded answer from those chunks. See `analysis_report.md` for findings and limitations.""")

nb["cells"]=C
nb.metadata["kernelspec"]={"display_name":"Python 3","language":"python","name":"python3"}
nbf.write(nb,"rag_pipeline.ipynb")
print("ok")
