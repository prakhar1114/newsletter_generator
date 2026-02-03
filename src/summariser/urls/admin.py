from __future__ import annotations

import asyncio
import secrets
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import HTMLResponse

from summariser.ingest import ingest_url, ingest_all_from_urls_path

admin_router = APIRouter(tags=["admin"])


@admin_router.get("/", response_class=HTMLResponse)
async def admin_page() -> HTMLResponse:
    # Note: app.py already mounts this router at prefix="/admin",
    # so this endpoint is served at GET /admin
    html = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Admin</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; }
      .card { max-width: 760px; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
      .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
      input[type="text"] { flex: 1; min-width: 280px; padding: 10px 12px; border: 1px solid #d1d5db; border-radius: 10px; }
      input[type="file"] { padding: 8px 0; }
      button { padding: 10px 14px; border: 1px solid #d1d5db; background: #111827; color: white; border-radius: 10px; cursor: pointer; }
      button.secondary { background: white; color: #111827; }
      pre { background: #0b1020; color: #e5e7eb; padding: 12px; border-radius: 12px; overflow: auto; }
      .muted { color: #6b7280; font-size: 14px; }
      h2 { margin: 0 0 12px 0; font-size: 18px; }
      label { font-weight: 600; }
    </style>
  </head>
  <body>
    <h1>Admin</h1>
    <p class="muted">Utilities for generating reports and managing sources.</p>

    <div class="card">
      <h2>Generate Report</h2>
      <div class="row">
        <div style="flex: 1; min-width: 280px;">
          <label for="jsonlFile">Select a jsonl file</label><br />
          <input id="jsonlFile" type="file" accept=".jsonl,application/jsonl,text/plain,application/json" />
        </div>
        <div>
          <button id="generateReportBtn">Generate Report</button>
        </div>
      </div>
      <p class="muted">Calls <code>/admin/generate_report</code> (dummy API for now).</p>
    </div>

    <div class="card">
      <h2>Add Urls</h2>
      <div class="row">
        <input id="urlsInput" type="text" placeholder="https://a.com, https://b.com" />
        <button id="addUrlsBtn" class="secondary">Add Urls</button>
      </div>
      <p class="muted">Calls <code>/admin/add_sources</code> (dummy API for now).</p>
    </div>

    <div class="card">
      <h2>Response</h2>
      <pre id="output">{}</pre>
    </div>

    <script>
      const output = document.getElementById("output");

      function setOutput(obj) {
        output.textContent = JSON.stringify(obj, null, 2);
      }

      document.getElementById("generateReportBtn").addEventListener("click", async () => {
        try {
          const fileInput = document.getElementById("jsonlFile");
          const file = fileInput.files && fileInput.files[0];

          const fd = new FormData();
          if (file) fd.append("file", file);

          const res = await fetch("/admin/generate_report", { method: "POST", body: fd });
          const data = await res.json().catch(() => ({ ok: false, error: "Non-JSON response" }));
          setOutput({ status: res.status, data });
        } catch (e) {
          setOutput({ ok: false, error: String(e) });
        }
      });

      document.getElementById("addUrlsBtn").addEventListener("click", async () => {
        try {
          const urls = document.getElementById("urlsInput").value || "";
          const res = await fetch("/admin/add_sources", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ urls })
          });
          const data = await res.json().catch(() => ({ ok: false, error: "Non-JSON response" }));
          setOutput({ status: res.status, data });
        } catch (e) {
          setOutput({ ok: false, error: String(e) });
        }
      });
    </script>
  </body>
</html>
""".strip()
    return HTMLResponse(content=html)


@admin_router.post("/generate_report")
async def generate_report(
    file: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    """
    Dummy endpoint: accepts an optional uploaded JSONL file.
    """
    filename = file.filename if file else None

    task_id = secrets.token_hex(8)

    # Run truly in the background. (FastAPI BackgroundTasks is sync-oriented;
    # passing an async fn there won't be awaited.)
    asyncio.create_task(ingest_all_from_urls_path(workers=4, limit=10))

    return {
        "ok": True,
        "message": "Report generation started in background.",
        "task_id": task_id,
        "filename": filename,
    }


async def _add_sources(urls: list[str]) -> None:
    # Placeholder for real persistence / ingestion later.
    for url in urls:
        await ingest_url(url)


@admin_router.post("/add_sources")
async def add_sources(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Dummy endpoint: accepts `{"urls": "comma,separated,urls"}` and parses it.
    """
    raw = str(payload.get("urls", "")).strip()
    urls = [u.strip() for u in raw.split(",") if u.strip()]
    await _add_sources(urls)
    return {"ok": True, "added_count": len(urls), "urls": urls}
