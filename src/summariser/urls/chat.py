from __future__ import annotations

import asyncio
import datetime as dt
import logging
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel, Field

from summariser.config import COLLECTION_NAME, REPORTS_PATH
from summariser.utils import fetch_markdown_from_id
from summariser.vectordb_client import client, init_vector_db

logger = logging.getLogger(__name__)

chat_router = APIRouter(tags=["chat"])


# In-memory session store: session_id -> previous_response_id
_SESSION_PREV_RESPONSE: dict[str, str] = {}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    ok: bool
    session_id: str
    response_id: str
    answer: str
    retrieved_count: int
    report_filename: str


_REPORT_STEM_RE = re.compile(r"^\d{8}_\d{4}$")


def _latest_report_path(reports_dir: Path = REPORTS_PATH) -> Path:
    if not reports_dir.exists():
        raise HTTPException(status_code=404, detail=f"Reports directory not found: {reports_dir}")
    files = sorted(reports_dir.glob("*.md"))
    if not files:
        raise HTTPException(status_code=404, detail=f"No report files found in {reports_dir}")

    ts_named = [p for p in files if _REPORT_STEM_RE.match(p.stem)]
    if ts_named:
        return max(ts_named, key=lambda p: p.stem)
    return max(files, key=lambda p: p.stat().st_mtime)


_embedder: Any | None = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        # Imported lazily so FastAPI can start without torch in some environments.
        from sentence_transformers import SentenceTransformer  # type: ignore

        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder


async def _embed_query(text: str) -> list[float]:
    def _encode() -> list[float]:
        emb = _get_embedder().encode(text, normalize_embeddings=True)
        return [float(x) for x in emb]

    return await asyncio.to_thread(_encode)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n\n[...truncated...]\n"


def _build_context(*, report_md: str, retrieved: list[dict[str, str]]) -> str:
    blocks: list[str] = []
    blocks.append("## Compiled report (latest)\n\n" + report_md)

    if retrieved:
        blocks.append("## Retrieved source articles (top matches)\n")
        for r in retrieved:
            blocks.append(
                "\n".join(
                    [
                        f"source: {r.get('url','')}",
                        f"file_id: {r.get('file_id','')}",
                        r.get("markdown", ""),
                    ]
                )
            )
    return "\n\n---\n\n".join(blocks)


@chat_router.get("/chat", response_class=HTMLResponse)
async def chat_page() -> HTMLResponse:
    html = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Chat</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; color: #111827; }
      .container { max-width: 980px; margin: 0 auto; }
      .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
      .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
      textarea { width: 100%; min-height: 90px; padding: 10px 12px; border: 1px solid #d1d5db; border-radius: 10px; font-family: inherit; }
      button { padding: 10px 14px; border: 1px solid #d1d5db; background: #111827; color: white; border-radius: 10px; cursor: pointer; }
      button.secondary { background: white; color: #111827; }
      .muted { color: #6b7280; font-size: 14px; }
      .error { color: #b91c1c; }
      .msg { padding: 10px 12px; border-radius: 12px; margin: 10px 0; white-space: pre-wrap; }
      .user { background: #f3f4f6; }
      .assistant { background: #eef2ff; }
      code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Chat with the report</h1>
      <p class="muted">Answers are constrained to the latest compiled report + top matching source articles.</p>

      <div class="card">
        <div class="row">
          <button id="newSessionBtn" class="secondary">New session</button>
          <span id="status" class="muted"></span>
        </div>
        <div class="muted" id="sessionMeta" style="margin-top:8px;"></div>
      </div>

      <div class="card">
        <label for="msg"><strong>Question</strong></label>
        <textarea id="msg" placeholder="Ask a question about the latest report..."></textarea>
        <div class="row" style="margin-top: 10px;">
          <button id="sendBtn">Send</button>
        </div>
      </div>

      <div class="card">
        <h2 style="margin-top:0;">Conversation</h2>
        <div id="thread"></div>
      </div>
    </div>

    <script>
      const statusEl = document.getElementById("status");
      const sessionMetaEl = document.getElementById("sessionMeta");
      const threadEl = document.getElementById("thread");
      const msgEl = document.getElementById("msg");
      const sendBtn = document.getElementById("sendBtn");
      const newSessionBtn = document.getElementById("newSessionBtn");

      let sessionId = localStorage.getItem("summariser_chat_session_id") || "";

      function setStatus(text, isError=false) {
        statusEl.textContent = text || "";
        statusEl.className = isError ? "muted error" : "muted";
      }

      function renderSession() {
        sessionMetaEl.textContent = sessionId ? ("session_id: " + sessionId) : "session_id: (new)";
      }

      function addMsg(role, text) {
        const div = document.createElement("div");
        div.className = "msg " + (role === "user" ? "user" : "assistant");
        div.textContent = (role === "user" ? "You: " : "Assistant: ") + text;
        threadEl.appendChild(div);
        div.scrollIntoView({ behavior: "smooth", block: "end" });
      }

      newSessionBtn.addEventListener("click", () => {
        sessionId = "";
        localStorage.removeItem("summariser_chat_session_id");
        setStatus("Started new session.");
        renderSession();
      });

      async function send() {
        const message = (msgEl.value || "").trim();
        if (!message) return;
        addMsg("user", message);
        msgEl.value = "";
        setStatus("Thinking...");
        sendBtn.disabled = true;
        try {
          const res = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, session_id: sessionId || null })
          });
          const data = await res.json().catch(() => null);
          if (!res.ok || !data || !data.ok) {
            throw new Error((data && (data.error || data.detail)) || ("HTTP " + res.status));
          }
          sessionId = data.session_id || sessionId;
          localStorage.setItem("summariser_chat_session_id", sessionId);
          renderSession();
          addMsg("assistant", String(data.answer || ""));
          setStatus("Done.");
        } catch (e) {
          setStatus("Error: " + String(e), true);
        } finally {
          sendBtn.disabled = false;
        }
      }

      sendBtn.addEventListener("click", send);
      msgEl.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") send();
      });

      renderSession();
    </script>
  </body>
</html>
""".strip()
    return HTMLResponse(content=html)


@chat_router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Chat endpoint that constrains answers to:
      - latest compiled report markdown
      - up to 10 vector-db matches for the query (score >= 0.7)

    Uses OpenAI Responses API and persists previous_response_id in memory by session_id.
    """
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must be non-empty")

    session_id = (req.session_id or "").strip() or secrets.token_hex(8)
    prev_response_id = _SESSION_PREV_RESPONSE.get(session_id)

    # Load latest report
    report_path = _latest_report_path()
    report_md = report_path.read_text(encoding="utf-8")

    # Embed query and retrieve top matches
    init_vector_db(COLLECTION_NAME)
    qvec = await _embed_query(message)

    qres = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,
        limit=50,
        with_payload=True,
        with_vectors=False,
    )
    threshold = 0.7
    matches = []
    for p in (qres.points or []):
        score = float(getattr(p, "score", 0.0) or 0.0)
        if score < threshold:
            continue
        payload = p.payload or {}
        file_id = str(payload.get("file_id", "")).strip()
        url = str(payload.get("url", "")).strip()
        if not file_id:
            continue
        matches.append((score, file_id, url))
    matches.sort(key=lambda x: x[0], reverse=True)
    matches = matches[:10]

    retrieved: list[dict[str, str]] = []
    for score, file_id, url in matches:
        try:
            md = fetch_markdown_from_id(file_id)
        except Exception as e:
            logger.info("[chat] failed to load markdown file_id=%s err=%r", file_id, e)
            continue
        retrieved.append(
            {
                "file_id": file_id,
                "url": url,
                "score": f"{score:.3f}",
                "markdown": _truncate(md, 4000),
            }
        )

    logger.info(
        "[chat] session=%s prev_response_id=%s retrieved=%s report=%s",
        session_id,
        prev_response_id or "-",
        len(retrieved),
        report_path.name,
    )

    context = _build_context(report_md=_truncate(report_md, 20000), retrieved=retrieved)

    instructions = (
        "You are chatting with a user about a compiled report. "
        "You MUST answer using ONLY the provided context (compiled report + retrieved sources). "
        "If the context does not contain the answer, reply exactly with: Data not available. "
        "Do not guess, do not use outside knowledge. "
        "When you use information from the context, cite the source URL if present."
    )

    # Responses API call with long timeout (~10 minutes)
    oai = OpenAI(timeout=600)
    resp = oai.responses.create(
        model="gpt-5-mini",
        instructions=instructions,
        previous_response_id=prev_response_id,
        input=(
            "Context:\n\n"
            f"{context}\n\n"
            "User question:\n"
            f"{message}\n"
        ),
      )

    answer = (getattr(resp, "output_text", "") or "").strip()
    response_id = str(getattr(resp, "id", "") or "")
    if not response_id:
        # Fallback: still return something usable
        response_id = secrets.token_hex(12)

    _SESSION_PREV_RESPONSE[session_id] = response_id
    if not answer:
        answer = "Data not available."

    return ChatResponse(
        ok=True,
        session_id=session_id,
        response_id=response_id,
        answer=answer,
        retrieved_count=len(retrieved),
        report_filename=report_path.name,
    )
