from __future__ import annotations

import asyncio
import json
import re
import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from crawl4ai import AsyncWebCrawler

from summariser.config import MARKDOWN_PATH, URLS_PATH
from summariser.vectordb_client import delete_from_db, init_vector_db, save_to_db, url_exists

@dataclass(frozen=True)
class IngestResult:
    url: str
    file_id: str
    point_id: str
    markdown_path: str
    markdown_bytes: int
    embedded: bool
    upserted: bool
    timings_ms: dict[str, float]


class IngestError(RuntimeError):
    pass


def iter_urls_from_jsonl(urls_path: Path) -> Iterable[str]:
    """
    Yields the `link` field from a JSONL file (one JSON object per line).
    Skips blank/invalid lines.
    """
    with urls_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = str(obj.get("link", "")).strip()
            if url:
                yield url


def generate_file_id6() -> str:
    # 6 characters, random. We'll collision-check against filesystem.
    return secrets.token_hex(3)


def _final_markdown_path(markdown_dir: Path, file_id: str) -> Path:
    return markdown_dir / f"{file_id}.md"

_HTTP_URL_RE = re.compile(r"https?://[^\s)]+")


def strip_http_urls(markdown: str) -> str:
    """
    Remove http/https URLs from markdown (simple regex-based sanitization).
    Example: `[](https://example.com/foo)` -> `[]()`
    """
    return _HTTP_URL_RE.sub("", markdown)


def write_markdown_temp(markdown_dir: Path, file_id: str, markdown: str) -> Path:
    """
    Writes markdown to a temp file in the same directory so the final move is atomic.
    Returns the temp path.
    """
    markdown_dir.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(4)
    tmp_path = markdown_dir / f"{file_id}.md.tmp-{nonce}"
    tmp_path.write_text(markdown, encoding="utf-8")
    return tmp_path


async def crawl_to_markdown(crawler: AsyncWebCrawler, url: str) -> str:
    """
    Crawl `url` and return markdown suitable for embedding.

    Uses Crawl4AI v0.8+ "fit markdown" with a pruning content filter when available
    (reduces boilerplate/nav/link-farms). Falls back to default crawling behavior
    if the relevant Crawl4AI classes aren't importable for any reason.
    """

    # Build an optional Crawl4AI config to generate fit markdown (v0.8+).
    config = None
    try:
        from crawl4ai import CrawlerRunConfig  # type: ignore
        from crawl4ai.content_filter_strategy import PruningContentFilter  # type: ignore
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator  # type: ignore

        prune_filter = PruningContentFilter(
            # Lower -> more content retained; higher -> more aggressively pruned
            threshold=0.7,
            threshold_type="dynamic",
            min_word_threshold=10,
        )
        md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)
        config = CrawlerRunConfig(markdown_generator=md_generator)
    except Exception:
        config = None

    # Crawl can hang on some sites (bot checks, long loads). Apply a timeout so we can skip.
    try:
        if config is None:
            result = await asyncio.wait_for(crawler.arun(url=url), timeout=45.0)
            print(result)
        else:
            result = await asyncio.wait_for(crawler.arun(url=url, config=config), timeout=45.0)
    except TimeoutError as e:
        raise IngestError(f"Crawl timed out after 45s for url={url}") from e

    # Crawl4AI can return an explicit error/success marker on some versions.
    success = getattr(result, "success", None)
    if success is False:
        msg = str(getattr(result, "error_message", "") or "Crawl4AI crawl failed")
        raise IngestError(msg)

    # Crawl4AI returns markdown in different shapes across versions; handle common cases.
    md = getattr(result, "markdown", None)
    if md is None:
        raise IngestError("Crawl4AI result has no markdown")

    # Older versions / some configs return markdown directly as a string.
    if isinstance(md, str):
        if not md.strip():
            raise IngestError("Crawl4AI markdown is empty")
        return md

    # v0.8+: markdown can be a MarkdownGenerationResult with raw_markdown/fit_markdown.
    # Prefer fit output first; fall back to raw.
    fit = getattr(md, "fit_markdown", None)
    if isinstance(fit, str) and fit.strip():
        return fit

    raw = getattr(md, "raw_markdown", None)
    if isinstance(raw, str) and raw.strip():
        return raw

    # Fallback to string conversion (last resort).
    text = str(md)
    if not text.strip():
        raise IngestError("Crawl4AI markdown is empty")
    return text


def _load_embedder():
    # Imported lazily so the FastAPI app can start without torch installed.
    # Reduce HuggingFace/Transformers log spam in notebooks/scripts.
    try:
        from transformers.utils import logging as hf_logging  # type: ignore

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass

    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


async def embed_text(text: str, *, embedder: Any) -> list[float]:
    # SentenceTransformer.encode is blocking; run it in a thread.
    def _encode() -> list[float]:
        vec = embedder.encode(text, normalize_embeddings=True)
        # vec can be numpy array; convert to plain list[float]
        return [float(x) for x in vec]

    return await asyncio.to_thread(_encode)


async def atomic_commit(
    *,
    url: str,
    file_id: str,
    markdown_temp_path: Path,
    embedding: list[float],
) -> tuple[Path, str]:
    """
    Best-effort atomic commit between filesystem markdown persistence and Qdrant upsert.
    """
    markdown_dir = markdown_temp_path.parent
    final_path = _final_markdown_path(markdown_dir, file_id)
    payload = {
        "url": url,
        "file_id": file_id,
        # Incremental/reporting metadata defaults:
        "cluster_id": -1,
        "is_centroid": False,
    }
    # Qdrant Local expects int or UUID-like ids. We generate a UUID id and keep
    # the human-friendly `file_id` in the payload + markdown filename.
    point_id = uuid.uuid4().hex

    # Ensure the collection exists (safe even when already created).
    await asyncio.to_thread(init_vector_db)
    try:
        await asyncio.to_thread(save_to_db, point_id, embedding, payload)
    except Exception:
        try:
            markdown_temp_path.unlink(missing_ok=True)
        finally:
            raise

    try:
        # Commit markdown file last, after vector is durable.
        markdown_temp_path.replace(final_path)
    except Exception:
        # Roll back the vector, then cleanup the temp file.
        try:
            await asyncio.to_thread(delete_from_db, point_id)
        finally:
            markdown_temp_path.unlink(missing_ok=True)
        raise

    return final_path, point_id


async def ingest_url(
    url: str,
    *,
    crawler: AsyncWebCrawler | None = None,
    embedder: Any | None = None,
    markdown_dir: Path = MARKDOWN_PATH,
    crawl_timeout_s: float = 45.0,
    embed_timeout_s: float = 60.0,
) -> IngestResult:
    """
    Public, reusable entrypoint for ingesting a single URL end-to-end.
    """
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    close_crawler = False

    if embedder is None:
        embedder = _load_embedder()

    # Generate unique file_id (best-effort collision avoidance against existing files)
    for _ in range(50):
        file_id = generate_file_id6()
        if not _final_markdown_path(markdown_dir, file_id).exists():
            break
    else:
        raise IngestError("Failed to generate unique 6-char file_id after 50 attempts")

    if crawler is None:
        crawler = AsyncWebCrawler()
        # AsyncWebCrawler should be used as a context manager; enter it here if we own it.
        await crawler.__aenter__()
        close_crawler = True

    try:
        t_crawl0 = time.perf_counter()
        # Apply timeout here as well (allows override per call).
        try:
            markdown = await asyncio.wait_for(crawl_to_markdown(crawler, url), timeout=crawl_timeout_s + 5.0)
        except TimeoutError as e:
            raise IngestError(f"Crawl timed out after {crawl_timeout_s}s for url={url}") from e
        markdown = strip_http_urls(markdown)
        timings["crawl_ms"] = (time.perf_counter() - t_crawl0) * 1000.0

        t_write0 = time.perf_counter()
        temp_path = write_markdown_temp(markdown_dir, file_id, markdown)
        timings["write_temp_ms"] = (time.perf_counter() - t_write0) * 1000.0

        try:
            t_embed0 = time.perf_counter()
            try:
                embedding = await asyncio.wait_for(embed_text(markdown, embedder=embedder), timeout=embed_timeout_s)
            except TimeoutError as e:
                raise IngestError(f"Embedding timed out after {embed_timeout_s}s for url={url}") from e
            timings["embed_ms"] = (time.perf_counter() - t_embed0) * 1000.0

            t_commit0 = time.perf_counter()
            final_path, point_id = await atomic_commit(
                url=url,
                file_id=file_id,
                markdown_temp_path=temp_path,
                embedding=embedding,
            )
            timings["commit_ms"] = (time.perf_counter() - t_commit0) * 1000.0
        except Exception:
            # If we fail after writing the temp markdown but before commit cleans it up,
            # remove the temp file to avoid orphaned `.md.tmp-*` artifacts.
            temp_path.unlink(missing_ok=True)
            raise

        timings["total_ms"] = (time.perf_counter() - t0) * 1000.0
        return IngestResult(
            url=url,
            file_id=file_id,
            point_id=point_id,
            markdown_path=str(final_path),
            markdown_bytes=final_path.stat().st_size,
            embedded=True,
            upserted=True,
            timings_ms=timings,
        )
    finally:
        if close_crawler:
            await crawler.__aexit__(None, None, None)  # type: ignore[misc]


async def ingest_all_from_urls_path(
    *,
    workers: int = 4,
    limit: int | None = None,
    urls_path: Path = URLS_PATH,
    markdown_dir: Path = MARKDOWN_PATH,
    crawl_timeout_s: float = 45.0,
    embed_timeout_s: float = 60.0,
) -> list[IngestResult]:
    """
    Batch ingestion from JSONL with a simple worker pool.

    Important: Crawl4AI's `AsyncWebCrawler` is not safe to share across concurrent
    tasks, so each worker owns its own crawler instance.
    """
    urls = list(iter_urls_from_jsonl(urls_path))
    if limit is not None:
        urls = urls[:limit]

    # Skip URLs already present in the vector DB (by payload match on `url`).
    if urls:
        await asyncio.to_thread(init_vector_db)
        filtered: list[str] = []
        for u in urls:
            if not await asyncio.to_thread(url_exists, u):
                filtered.append(u)
        urls = filtered

    queue: asyncio.Queue[str] = asyncio.Queue()
    for u in urls:
        queue.put_nowait(u)

    results: list[IngestResult] = []
    failures: list[dict[str, str]] = []

    async def worker() -> None:
        embedder = _load_embedder()
        async with AsyncWebCrawler() as crawler:
            while True:
                try:
                    u = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    res = await ingest_url(
                        u,
                        crawler=crawler,
                        embedder=embedder,
                        markdown_dir=markdown_dir,
                        crawl_timeout_s=crawl_timeout_s,
                        embed_timeout_s=embed_timeout_s,
                    )
                except Exception as e:
                    failures.append({"url": u, "error": repr(e)})
                    continue
                else:
                    results.append(res)

    worker_count = max(1, min(workers, len(urls) if urls else 1))
    await asyncio.gather(*[worker() for _ in range(worker_count)])

    if failures:
        markdown_dir.mkdir(parents=True, exist_ok=True)
        errors_path = markdown_dir / "ingest_errors.jsonl"
        with errors_path.open("a", encoding="utf-8") as f:
            for row in failures:
                f.write(json.dumps(row) + "\n")

    return results
