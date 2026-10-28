"""Lightweight Agentic RAG web demo for Render Free.

Groq hosts the Router and Answer Agent calls. PDF retrieval uses a small local
TF-IDF sparse-vector index (NOT neural semantic embeddings, ChromaDB or ONNX).
This web adaptation intentionally does not execute the original CrewAI notebook.
"""

import asyncio
import html
import json
import logging
import math
import os
import re
import time
import uuid
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentic-rag")

APP_VERSION = "3.1-progress-502-fix"
APP_DIR = Path(__file__).resolve().parent
PDF_PATH = APP_DIR / "trasformer_research_paper-dataset.pdf"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").removeprefix("groq/")
# In-memory jobs: suitable for a single-worker demo. They will be lost on restart.
JOBS: dict[str, dict] = {}
ACTIVE = set()
MAX_JOBS = 40

app = FastAPI(title="Agentic RAG — Lightweight Render Demo", version=APP_VERSION)


class QuestionInput(BaseModel):
    question: str = Field(min_length=3, max_length=1500)


def progress(job_id: str, stage: str, detail: str, **fields):
    job = JOBS.get(job_id)
    if job is None:
        return
    job.update(stage=stage, detail=detail, **fields)
    logger.info("job=%s stage=%s %s", job_id, stage, detail)


async def groq_chat(messages: list[dict], max_tokens: int = 1500) -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing in Render Environment.")
    body = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_completion_tokens": max_tokens,
        "include_reasoning": False,
        "reasoning_effort": "low",
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=10.0)) as client:
            response = await client.post(
                GROQ_ENDPOINT,
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
            )
            response.raise_for_status()
            data = response.json()
        content = data["choices"][0]["message"].get("content") or ""
        if not content.strip():
            raise RuntimeError("Groq returned an empty answer. Try your question again.")
        return content.strip()
    except httpx.TimeoutException as exc:
        raise RuntimeError("Groq response timed out. Please retry.") from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 401:
            detail = "GROQ_API_KEY was rejected. Check the key in Render Environment."
        elif status == 429:
            detail = "Groq rate limit reached. Wait a little before trying again."
        else:
            detail = f"Groq request failed (HTTP {status}). Check Render logs."
            logger.warning("Groq HTTP %s: %s", status, exc.response.text[:600])
        raise RuntimeError(detail) from exc
    except httpx.RequestError as exc:
        raise RuntimeError("Could not connect to Groq. Check internet connectivity and retry.") from exc


async def route_question(question: str) -> tuple[str, str]:
    messages = [
        {"role": "system", "content": (
            "You are the Router Agent for a RAG application. Choose only one source. "
            "PDF: questions about the provided Attention Is All You Need Transformer "
            "research paper (attention, encoder/decoder, positional encoding, results, "
            "architecture) or the uploaded PDF. "
            "WEB: requests for up-to-date facts, current events, new work, or topics "
            "that need current online sources. "
            "GENERAL: unrelated everyday or creative questions. "
            "Return only a JSON object with keys route and reason. "
            "route must be PDF, WEB or GENERAL, with a short reason."
        )},
        {"role": "user", "content": question},
    ]
    raw = await groq_chat(messages, max_tokens=300)
    try:
        match = re.search(r"\{.*?\}", raw, flags=re.S)
        decision = json.loads(match.group(0) if match else raw)
        route = str(decision["route"]).upper().strip()
        if route not in ("PDF", "WEB", "GENERAL"):
            raise ValueError("Bad route")
        return route, str(decision.get("reason", "Router selected this source."))[:250]
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        # Safe, explicit fallback to avoid a never-ending agent retry loop.
        tokens = set(tokenize(question))
        if tokens & {"pdf", "transformer", "encoder", "decoder", "attention", "vaswani", "positional", "stack"}:
            return "PDF", "Fallback: question concerns the research paper."
        if tokens & {"latest", "recent", "today", "current", "2026", "news", "this", "week"}:
            return "WEB", "Fallback: question asks for recent information."
        return "GENERAL", "Fallback: general question."


STOPWORDS = set("a an and are as at be by can did do does for from has have how i in into is it its of on or our that the their these they this to was were what when where which who why with you your about explain describe give me tell much more than paper".split())


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


def make_chunks(text: str, page_num: int):
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Short overlapping windows retain section context and simplify citations.
    for start in range(0, len(text), 850):
        snippet = text[start:start + 1200].strip()
        if len(snippet) > 80:
            yield {"page": page_num, "text": snippet}


@lru_cache(maxsize=1)
def pdf_index():
    if not PDF_PATH.is_file():
        raise RuntimeError(f"Missing {PDF_PATH.name} in the GitHub repository root.")
    reader = PdfReader(str(PDF_PATH))
    chunks = []
    for page_num, page in enumerate(reader.pages, 1):
        for chunk in make_chunks(page.extract_text() or "", page_num):
            chunks.append(chunk)
    if not chunks:
        raise RuntimeError("The PDF has no extractable text.")
    doc_counts = [Counter(tokenize(c["text"])) for c in chunks]
    df = Counter()
    for counts in doc_counts:
        df.update(counts.keys())
    n = len(chunks)
    weights = []
    norms = []
    for counts in doc_counts:
        w = {t: (1 + math.log(count)) * math.log(1 + n / (1 + df[t]))
             for t, count in counts.items()}
        weights.append(w)
        norms.append(math.sqrt(sum(v * v for v in w.values())) or 1)
    logger.info("PDF index ready: %d pages, %d chunks", len(reader.pages), len(chunks))
    return chunks, weights, norms, df


def search_pdf(question: str, limit: int = 5) -> list[dict]:
    chunks, weights, norms, df = pdf_index()
    query_counts = Counter(tokenize(question))
    if not query_counts:
        return []
    n = len(chunks)
    query_w = {t: (1 + math.log(count)) * math.log(1 + n / (1 + df.get(t, 0)))
               for t, count in query_counts.items()}
    qnorm = math.sqrt(sum(v * v for v in query_w.values())) or 1
    candidates = []
    for idx, document in enumerate(weights):
        dot = sum(value * document.get(term, 0) for term, value in query_w.items())
        if dot:
            candidates.append((dot / (norms[idx] * qnorm), idx))
    candidates.sort(reverse=True)
    top = []
    for score, idx in candidates:
        # Avoid repeated near-identical slices on the same page.
        if score <= 0.01:
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = round(score, 3)
        top.append(chunk)
        if len(top) == limit:
            break
    return top


async def web_search(question: str) -> list[dict]:
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        raise RuntimeError("WEB route needs TAVILY_API_KEY in Render Environment.")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(22.0, connect=8.0)) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {key}"},
                json={"query": question, "search_depth": "basic", "max_results": 5},
            )
            resp.raise_for_status()
            return (resp.json().get("results") or [])[:5]
    except httpx.TimeoutException as exc:
        raise RuntimeError("Web search timed out. Please retry.") from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Tavily web search failed (HTTP {exc.response.status_code}).") from exc
    except httpx.RequestError as exc:
        raise RuntimeError("Could not connect to Tavily search.") from exc


async def handle_question(job_id: str, question: str):
    ACTIVE.add(job_id)
    try:
        progress(job_id, "routing", "Router Agent is selecting PDF, WEB or GENERAL…")
        route, reason = await route_question(question)
        progress(job_id, "retrieval", f"Router selected {route}. Preparing evidence…", route=route, reason=reason)
        evidence = ""
        sources = []
        if route == "PDF":
            progress(job_id, "retrieval", "Searching the PDF vector index for relevant passages…")
            snippets = await asyncio.to_thread(search_pdf, question)
            if not snippets:
                progress(job_id, "done", "No matching PDF passages found.",
                         answer="I couldn't find supporting text in the uploaded PDF for this question.", sources=[])
                return
            sources = [f"PDF page {chunk['page']}" for chunk in snippets]
            evidence = "\n\n".join(
                f"[PDF page {c['page']}]\n{c['text']}" for c in snippets
            )
        elif route == "WEB":
            progress(job_id, "retrieval", "Searching the web with Tavily…")
            results = await web_search(question)
            if not results:
                progress(job_id, "done", "No web search results were found.",
                         answer="No relevant web results were returned. Please try a different question.", sources=[])
                return
            sources = [r.get("url", "") for r in results if r.get("url")]
            evidence = "\n\n".join(
                f"Source {i}: {r.get('title','Untitled')}\nURL: {r.get('url','')}\n"
                f"Evidence: {str(r.get('content',''))[:1600]}"
                for i, r in enumerate(results, 1)
            )
        progress(job_id, "generating", "Retriever Agent is writing an evidence-grounded answer…")
        if route == "PDF":
            system = (
                "You are a careful PDF-based Retriever Agent. Answer ONLY using the supplied "
                "excerpts from the research paper Attention Is All You Need. "
                "Cite pages as [PDF page N] using only page numbers shown in the excerpts. "
                "If the evidence is insufficient, clearly say so. Do not invent facts or citations."
            )
        elif route == "WEB":
            system = (
                "You are a research Retriever Agent. Answer ONLY using supplied Tavily search evidence. "
                "Include source URLs, distinguish publication dates if available, and admit uncertainty."
            )
        else:
            system = "You are a concise, practical assistant. Do not claim PDF or web evidence for general answers."
        user_text = f"Question: {question}\n\nRetrieved evidence:\n{evidence[:13000]}"
        answer = await groq_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user_text}],
            max_tokens=1800,
        )
        progress(job_id, "done", "Answer completed.", answer=answer, sources=sources)
    except Exception as exc:
        logger.exception("job=%s error", job_id)
        progress(job_id, "error", str(exc)[:350], error=str(exc)[:350])
    finally:
        ACTIVE.discard(job_id)


@app.get("/health")
async def health():
    return {"status": "ok", "pdf_found": PDF_PATH.is_file(), "mode": "lightweight", "version": APP_VERSION}


@app.post("/api/ask", status_code=202)
async def ask(data: QuestionInput):
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=503, detail="Set GROQ_API_KEY under Render Environment.")
    if ACTIVE:
        raise HTTPException(status_code=429, detail="Another question is running. Please try again once it finishes.")
    for stale in list(JOBS)[:-MAX_JOBS]:
        JOBS.pop(stale, None)
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"stage": "queued", "detail": "Question received.", "route": "", "answer": "", "sources": []}
    ACTIVE.add(job_id)  # set before creating task to prevent duplicate jobs
    asyncio.create_task(handle_question(job_id, data.question.strip()))
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def job_state(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found. The server may have restarted.")
    return JOBS[job_id]


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agentic RAG • Router / Retriever</title>
<style>
:root{color-scheme:light;font-family:ui-sans-serif,system-ui,"Segoe UI",Arial,sans-serif;color:#172c41}
*{box-sizing:border-box}body{background:#f3f7fc;margin:0;padding:24px}
main{max-width:860px;margin:26px auto;background:#fff;padding:30px;border:1px solid #e2eaf5;border-radius:18px;box-shadow:0 10px 35px #1234a10c}
h1{font-size:28px;margin:0 0 8px}.sub{color:#51657e;margin-top:0;line-height:1.6}
label{font-weight:650;display:block;margin:22px 0 8px}textarea{width:100%;height:125px;border:1px solid #c2cede;border-radius:10px;padding:13px;font:inherit;resize:vertical}
textarea:focus{border-color:#1464dc;outline:2px solid #b9d6ff}button{background:#155ed4;color:white;font:inherit;font-weight:650;border:0;padding:12px 30px;border-radius:10px;cursor:pointer;margin-top:13px}
button:disabled{opacity:.55;cursor:not-allowed}.small{color:#607084;font-size:13px}
#progress{display:block;padding:17px 18px;border:1px solid #dce7f7;background:#f7faff;border-radius:12px;margin-top:22px}
#stageText{font-weight:650}#timer{float:right;font-variant-numeric:tabular-nums;color:#5b6e81}
.bar{background:#dce6f5;height:8px;border-radius:999px;margin:13px 0 12px;overflow:hidden}
#barFill{width:0%;height:100%;border-radius:999px;background:#1b66d4;transition:width .5s ease}
.steps{display:flex;justify-content:space-between;color:#62758d;font-size:12px;gap:8px;flex-wrap:wrap}.steps .active{font-weight:700;color:#155ed4}
#output{display:none;border-top:1px solid #edf1f7;margin-top:23px;padding-top:18px}#answer{white-space:pre-wrap;line-height:1.7;overflow-wrap:anywhere}
#error{display:none;color:#a02125;background:#fff0f0;padding:12px;border-radius:9px;margin-top:16px;white-space:pre-wrap}
#route{font-size:13px;color:#20518a;font-weight:700;margin-bottom:8px}
@media(max-width:600px){body{padding:10px}main{margin:8px auto;padding:20px}.steps{font-size:11px}h1{font-size:24px}}
</style></head>
<body><main>
<h1>Agentic RAG — Router / Retriever</h1>
<p class="small" id="buildVersion">Build v3.1 · Live progress enabled · Lightweight PDF retrieval</p>
<p class="sub">Ask about the included <strong>Attention Is All You Need</strong> PDF, or ask a question requiring web search. Powered by Groq and a lightweight PDF vector index.</p>
<form id="askForm"><label for="question">Your question</label>
<textarea id="question" required minlength="3" maxlength="1500" placeholder="Explain the Encoder and Decoder Stacks from the PDF"></textarea>
<button id="askButton" type="submit">Ask</button></form>
<div id="progress" role="status" aria-live="polite"><span id="stageText">Ready — click Ask to start</span><span id="timer">00:00</span>
<div class="bar"><div id="barFill"></div></div><div class="steps"><span id="s1">1. Routing</span><span id="s2">2. Retrieval</span><span id="s3">3. Answer</span></div>
<p class="small" id="detail" style="margin-bottom:0">Your question will show live stages here: Routing → Retrieval → Answer.</p></div>
<div id="error" role="alert"></div><section id="output"><div id="route"></div><h2>Answer</h2><div id="answer"></div><div id="sources" class="small"></div></section>
<p class="small" style="margin-top:30px">Free Render instances may take longer to wake after inactivity. A 502 on the entire page means the server itself is unavailable; check the Render Logs.</p>
</main>
<script>
const form=document.getElementById('askForm'),btn=document.getElementById('askButton');
const progressBox=document.getElementById('progress'),stageText=document.getElementById('stageText'),timer=document.getElementById('timer'),bar=document.getElementById('barFill'),detail=document.getElementById('detail');
const output=document.getElementById('output'),errorBox=document.getElementById('error');
const stages={queued:[8,'Question received'],routing:[24,'Router Agent'],retrieval:[58,'Retrieving evidence'],generating:[85,'Generating answer'],done:[100,'Completed'],error:[100,'Error']};
function showError(message){errorBox.style.display='block';errorBox.textContent=message;}
function setStage(stage,message){const v=stages[stage]||stages.queued;bar.style.width=v[0]+'%';stageText.textContent=v[1];detail.textContent=message||'';
 ['s1','s2','s3'].forEach((id,i)=>document.getElementById(id).classList.toggle('active', (stage==='routing'&&i===0)||(stage==='retrieval'&&i===1)||(stage==='generating'&&i===2)));
}
function pause(ms){return new Promise(resolve=>setTimeout(resolve,ms));}
async function jsonResponse(url, options){let response;try{response=await fetch(url,options);}catch(e){throw new Error('Could not reach Render. The server may have stopped or restarted. Check Render Logs.');}
let data;try{data=await response.json();}catch(e){throw new Error('Render returned an invalid response (possibly a 502). Check Logs and /health.');}
if(!response.ok){throw new Error(typeof data.detail==='string'?data.detail:'HTTP '+response.status);}return data;}
form.addEventListener('submit',async(event)=>{event.preventDefault();
 const question=document.getElementById('question').value.trim();if(!question)return;
 btn.disabled=true;btn.textContent='Processing…';output.style.display='none';errorBox.style.display='none';progressBox.style.display='block';
 setStage('queued','Sending your question…');const started=Date.now();const tick=setInterval(()=>{const s=Math.floor((Date.now()-started)/1000);timer.textContent=String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');},1000);
 try{
 const {job_id}=await jsonResponse('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question})});
 let data=null;const stopAt=Date.now()+115000;
 while(Date.now()<stopAt){await pause(1200);data=await jsonResponse('/api/jobs/'+encodeURIComponent(job_id));setStage(data.stage,data.detail);
   if(data.stage==='done'||data.stage==='error')break;}
 if(!data||!['done','error'].includes(data.stage))throw new Error('This request exceeded the demo time limit. Check Render Logs or try a shorter question.');
 if(data.stage==='error')throw new Error(data.error||data.detail||'The task failed.');
 document.getElementById('route').textContent='Route: '+(data.route||'GENERAL')+(data.reason?' — '+data.reason:'');
 document.getElementById('answer').textContent=data.answer||'No answer returned.';
 document.getElementById('sources').textContent=data.sources?.length?'Evidence: '+[...new Set(data.sources)].join(' • '):'';
 output.style.display='block';
 }catch(e){setStage('error','An error occurred.');showError(e.message||String(e));}
 finally{clearInterval(tick);btn.disabled=false;btn.textContent='Ask';}
});
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
async def home():
    return PAGE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
