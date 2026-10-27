# Healthcare PDF RAG — Groq + FAISS + Gradio

Simple PDF retrieval-augmented generation (RAG) application, **not a multi-agent system**.
It indexes three fictional healthcare PDFs and lets you replace them with PDFs uploaded
by the current browser session. Answers are generated with Groq from retrieved passages
and show file/page citations. The deployed app defaults to `openai/gpt-oss-120b` because
the original `llama-3.3-70b-versatile` model was retired for free Groq accounts.

## Running locally (full sentence-transformer embeddings)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env to set GROQ_API_KEY — do not commit it!
.venv\Scripts\python app.py
```

## Deploy to Render free

Push **this `render_backend` folder's contents** to your own GitHub repository.
Create a Render Web Service using Python; use:
- Build command: `pip install -r requirements-render.txt`
- Start command: `python app.py`
- Compute plan: Free (512 MB RAM)
- Environment variables: `GROQ_API_KEY` (secret), `RAG_OFFLINE=1`, `PYTHON_VERSION=3.11.11`

Render free is memory-limited. `RAG_OFFLINE=1` uses the lightweight **TF-IDF/SVD fallback**
instead of downloading the heavier Hugging Face sentence-transformer model. This mode is
still a PDF -> chunks -> vector search -> Groq RAG pipeline, but the rankings may differ.
If memory or build limits persist, a paid service with more RAM or a different host is needed.

## Files

- `app.py` — Gradio chatbot (sample and uploaded PDFs, sources)
- `data/` — three fictional sample hospital PDFs
- `requirements.txt` — full local/notebook setup
- `requirements-render.txt` — lightweight deployment dependencies
- `.python-version` — pin Python 3.11 for Render
- `render.yaml` — optional Render Blueprint config
- `rag_pipeline.ipynb` / `rag_pipeline_executed.ipynb` — original notebooks
- `analysis_report.md`, `outputs/` — original study/evaluation files

**Privacy note:** This is a public demo and must not process real patient, medical or
confidential PDFs. Uploaded text is sent to Groq when the answer-generation checkbox is
on. Session indices are in-memory and ephemeral, not persistent; a production healthcare
app needs authentication, access control, consent, retention policies and auditing.
