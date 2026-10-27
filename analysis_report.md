# Analysis Report: RAG Pipeline over PDF Documents

## 1. Approach
Three fictional hospital documents for patients (admission and discharge guide, insurance and billing FAQ, outpatient and medication safety handbook; 6 pages total) were loaded with
LangChain's `PyPDFLoader`, split with `RecursiveCharacterTextSplitter` (400 characters, 80 overlap, 24 chunks),
embedded, and stored in a FAISS index. Queries are embedded the same way and the top 3 chunks are returned by L2 distance.
A Groq-hosted LLM (`llama-3.3-70b-versatile`) then answers using only the retrieved chunks and cites source and page.

**Deviation from the brief:** OpenAI embeddings were replaced because no OpenAI key was available and Groq offers no embeddings
endpoint. Embeddings use `sentence-transformers/all-MiniLM-L6-v2` locally; Groq handles generation.

A Gradio interface (`app.py`) wraps the same pipeline: the user types a question, sees the Groq answer and the retrieved source chunks with file, page and distance, and can upload other PDFs to rebuild the index.

## 2. Results (9 patient-style questions, 3 per document)
Figures below come from the offline TF-IDF/SVD fallback embedder, not the MiniLM model (see README). Re-run to refresh them.

| Metric | Result |
|---|---|
| Correct source document ranked first (hit@1) | 9/9 |
| Chunk containing the answer in top 1 | 6/9 |
| Chunk containing the answer in top 3 | 9/9 |

Chunk-size experiment (same questions):

| Chunk size / overlap | Chunks | Source hit@1 | Answer in top 1 | Answer in top 3 |
|---|---|---|---|---|
| 200 / 40 | 54 | 1.00 | 0.33 | 0.56 |
| 400 / 80 | 24 | 1.00 | 0.67 | 1.00 |
| 800 / 160 | 13 | 1.00 | 1.00 | 1.00 |

## 3. Findings
- **Document-level retrieval was easy.** Admission, billing and medication topics are clearly different, so the right source ranked first every time.
- **Chunk size mattered most.** At 200 characters, facts were separated from their context and only 33% of top results held the answer.
  At 800 characters every question was answered by the top chunk, but each chunk then mixes several policies, which would pass more irrelevant text to the LLM.
- **Top-1 misses were ranking issues, not missing content.** In the 400-character run, three appointment/test/side-effect questions had the answer in
  the 2nd or 3rd chunk, which is why top-3 retrieval (used for generation) is more reliable than top-1.
- **Evaluation bug found and fixed.** PDF line wraps (e.g. "7 calendar / days") caused false misses in keyword matching until whitespace was normalised.
- **Safety design for healthcare.** The Groq prompt restricts answers to retrieved text, requires citations, and tells the model not to give medical advice, redirecting clinical questions to a doctor.

## 4. Limitations
- The test set is small (9 questions, 3 short synthetic documents), so the numbers show behaviour, not benchmark accuracy.
- The keyword check measures whether a fact string is present, not whether the answer is complete.
- Results here use the fallback embedder; MiniLM should handle paraphrased questions better (this has not been measured here).
- No re-ranking or hybrid keyword search was used.

## 5. Possible improvements
Section-aware chunking (split on headings), 800-character chunks with higher overlap, a cross-encoder re-ranker, adding
MMR for diversity, and larger evaluation sets with human-judged answers.
