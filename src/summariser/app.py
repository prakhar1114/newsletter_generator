from __future__ import annotations

from contextlib import asynccontextmanager
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from summariser.config import REPORTS_PATH
from summariser.vectordb_client import init_vector_db
from summariser.urls.admin import admin_router # We will create this next

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    """
    # 1. Initialize Qdrant Collection
    print("Checking Vector DB collections...")
    init_vector_db()

    # 2. You could also initialize SQL DBs or LLM clients here

    yield
    # Shutdown logic goes here (e.g., closing DB connections)

def create_app() -> FastAPI:
    app = FastAPI(
        title="Summariser AI API",
        version="0.1.0",
        lifespan=lifespan
    )

    # Standard Middleware (for frontend/different origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include different URL groups (Routers)
    app.include_router(admin_router, prefix="/admin")

    @app.get("/", response_class=HTMLResponse)
    async def root_page() -> HTMLResponse:
        """
        Minimal UI to fetch and view the latest report.
        """
        html = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Summariser</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; color: #111827; }
      .container { max-width: 980px; margin: 0 auto; }
      .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
      .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
      button { padding: 10px 14px; border: 1px solid #d1d5db; background: #111827; color: white; border-radius: 10px; cursor: pointer; }
      button.secondary { background: white; color: #111827; }
      .muted { color: #6b7280; font-size: 14px; }
      .meta { font-size: 14px; color: #374151; }
      .error { color: #b91c1c; }
      #rendered h1, #rendered h2, #rendered h3 { margin-top: 18px; }
      #rendered pre { background: #0b1020; color: #e5e7eb; padding: 12px; border-radius: 12px; overflow: auto; }
      #rendered code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
      #rendered a { color: #2563eb; text-decoration: underline; }
      #raw { white-space: pre-wrap; background: #f9fafb; padding: 12px; border-radius: 12px; border: 1px solid #e5e7eb; overflow: auto; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Summariser</h1>
      <p class="muted">Fetch and view the latest compiled report.</p>

      <div class="card">
        <div class="row">
          <button id="fetchBtn">Fetch Report</button>
          <button id="toggleBtn" class="secondary" disabled>Show raw markdown</button>
          <span id="status" class="muted"></span>
        </div>
        <div id="meta" class="meta" style="margin-top: 10px;"></div>
      </div>

      <div class="card">
        <div id="rendered"></div>
        <div id="raw" style="display:none;"></div>
      </div>
    </div>

    <script>
      const fetchBtn = document.getElementById("fetchBtn");
      const toggleBtn = document.getElementById("toggleBtn");
      const statusEl = document.getElementById("status");
      const metaEl = document.getElementById("meta");
      const renderedEl = document.getElementById("rendered");
      const rawEl = document.getElementById("raw");

      function escapeHtml(s) {
        return s
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#039;");
      }

      // Lightweight markdown renderer (enough for our reports).
      // Supports: headings (#/##/###), fenced code blocks, paragraphs, links, inline code, and basic lists.
      function renderMarkdown(md) {
        const lines = md.split(/\\r?\\n/);
        let html = "";
        let inCode = false;
        let listOpen = false;
        let codeBuf = [];

        function flushList() {
          if (listOpen) {
            html += "</ul>";
            listOpen = false;
          }
        }

      function inlineFormat(text) {
          // Protect explicit markdown links first so later URL autolinking doesn't
          // corrupt the href attribute.
          const links = [];
          const protectedText = text.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, (_m, label, url) => {
            const key = `@@LINK${links.length}@@`;
            links.push({ label, url });
            return key;
          });

          let s = escapeHtml(protectedText);

          // inline code
          s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
          // bold **text**
          s = s.replace(/\\*\\*([^*]+)\\*\\*/g, "<strong>$1</strong>");
          // italics *text* (simple)
          s = s.replace(/(^|\\s)\\*([^*]+)\\*(\\s|$)/g, "$1<em>$2</em>$3");

          // Auto-link bare URLs.
          s = s.replace(/\\bhttps?:\\/\\/[^\\s<]+/g, (u) => {
            // Trim common trailing punctuation
            const trimmed = u.replace(/[),.;!?]+$/, "");
            const trail = u.slice(trimmed.length);
            return `<a href="${trimmed}" target="_blank" rel="noreferrer">${trimmed}</a>${trail}`;
          });

          // Restore protected markdown links.
          for (let i = 0; i < links.length; i++) {
            const key = `@@LINK${i}@@`;
            const href = escapeHtml(String(links[i].url || ""));
            const label = escapeHtml(String(links[i].label || ""));
            s = s.replaceAll(key, `<a href="${href}" target="_blank" rel="noreferrer">${label}</a>`);
          }

          return s;
        }

        for (const line of lines) {
          if (line.trim().startsWith("```")) {
            if (!inCode) {
              flushList();
              inCode = true;
              codeBuf = [];
            } else {
              inCode = false;
              const code = escapeHtml(codeBuf.join("\\n"));
              html += "<pre><code>" + code + "</code></pre>";
              codeBuf = [];
            }
            continue;
          }

          if (inCode) {
            codeBuf.push(line);
            continue;
          }

          const t = line.trim();
          if (!t) {
            flushList();
            continue;
          }

          // headings
          if (t.startsWith("### ")) { flushList(); html += "<h3>" + inlineFormat(t.slice(4)) + "</h3>"; continue; }
          if (t.startsWith("## ")) { flushList(); html += "<h2>" + inlineFormat(t.slice(3)) + "</h2>"; continue; }
          if (t.startsWith("# ")) { flushList(); html += "<h1>" + inlineFormat(t.slice(2)) + "</h1>"; continue; }

          // unordered list
          if (t.startsWith("- ")) {
            if (!listOpen) { html += "<ul>"; listOpen = true; }
            html += "<li>" + inlineFormat(t.slice(2)) + "</li>";
            continue;
          }

          flushList();
          html += "<p>" + inlineFormat(t) + "</p>";
        }
        flushList();
        return html;
      }

      function setStatus(text, isError=false) {
        statusEl.textContent = text || "";
        statusEl.className = isError ? "muted error" : "muted";
      }

      let showingRaw = false;
      toggleBtn.addEventListener("click", () => {
        showingRaw = !showingRaw;
        rawEl.style.display = showingRaw ? "block" : "none";
        renderedEl.style.display = showingRaw ? "none" : "block";
        toggleBtn.textContent = showingRaw ? "Show rendered" : "Show raw markdown";
      });

      fetchBtn.addEventListener("click", async () => {
        setStatus("Fetching latest report...");
        fetchBtn.disabled = true;
        try {
          const res = await fetch("/latest_report");
          const data = await res.json().catch(() => null);
          if (!res.ok || !data || !data.ok) {
            throw new Error((data && (data.error || data.detail)) || ("HTTP " + res.status));
          }

          const md = String(data.markdown || "");
          metaEl.textContent = data.filename ? ("Latest: " + data.filename) : "";

          renderedEl.innerHTML = renderMarkdown(md);
          rawEl.textContent = md;
          toggleBtn.disabled = false;
          showingRaw = false;
          rawEl.style.display = "none";
          renderedEl.style.display = "block";
          toggleBtn.textContent = "Show raw markdown";
          setStatus("Done.");
        } catch (e) {
          setStatus("Failed to fetch report: " + String(e), true);
        } finally {
          fetchBtn.disabled = false;
        }
      });
    </script>
  </body>
</html>
""".strip()
        return HTMLResponse(content=html)

    _REPORT_STEM_RE = re.compile(r"^\\d{8}_\\d{4}$")

    @app.get("/latest_report")
    async def latest_report() -> dict[str, str | bool]:
        """
        Returns the latest report markdown from REPORTS_PATH.
        """
        reports_dir: Path = REPORTS_PATH
        if not reports_dir.exists():
            raise HTTPException(status_code=404, detail=f"Reports directory not found: {reports_dir}")

        files = sorted(reports_dir.glob("*.md"))
        if not files:
            raise HTTPException(status_code=404, detail=f"No report files found in {reports_dir}")

        # Prefer timestamped names like YYYYMMDD_HHMM.md (lexicographically sortable).
        ts_named = [p for p in files if _REPORT_STEM_RE.match(p.stem)]
        if ts_named:
            latest = max(ts_named, key=lambda p: p.stem)
        else:
            latest = max(files, key=lambda p: p.stat().st_mtime)

        md = latest.read_text(encoding="utf-8")
        return {"ok": True, "filename": latest.name, "markdown": md}

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "database": "connected"}

    return app

app = create_app()
