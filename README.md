# Newsletter Generator

Ingests thousands of articles from URLs, clusters them by topic using embedding similarity, and produces a consolidated newsletter report with source citations. Supports incremental updates — new articles are slotted into existing topic clusters without re-processing the full dataset. A chat interface allows RAG-based Q&A over the ingested corpus.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Web Server                        │
│         Report Viewer  /  Admin Panel  /  Chat UI           │
└────────────┬──────────────────────┬────────────────────────┘
             │                      │
      ┌──────▼──────┐       ┌───────▼────────┐
      │  Ingestion  │       │   Chat (RAG)   │
      │  Pipeline   │       │                │
      └──────┬──────┘       └───────┬────────┘
             │                      │
             ▼                      ▼
      ┌──────────────┐      ┌──────────────┐
      │   Crawl4AI   │      │  Vector      │
      │  (scraping)  │      │  Search      │
      └──────┬───────┘      └──────┬───────┘
             │                     │
             ▼                     │
      ┌──────────────┐             │
      │  Sentence    │             │
      │  Transformer │             │
      │  (384d emb)  │             │
      └──────┬───────┘             │
             │                     │
             ▼                     ▼
      ┌──────────────────────────────────┐
      │        Qdrant (local)            │
      │   vectors + cluster metadata     │
      └──────────────┬───────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   ┌────────────┐     ┌───────────────┐
   │  Clusterer │     │  OpenAI LLM   │
   │ UMAP+HDBSC│     │  (report gen) │
   └────────────┘     └───────────────┘
```

### Core Components

| Component | File(s) | Role |
|---|---|---|
| **Ingestion** | `src/summariser/ingest.py` | Crawls URLs, embeds content, persists to vector DB and disk |
| **Vector DB** | `src/summariser/vectordb_client.py` | Qdrant wrapper — upsert, search, scroll, filter by cluster/centroid |
| **Clustering & Reporting** | `src/summariser/reporting.py` | UMAP dimensionality reduction → HDBSCAN clustering → centroid selection → LLM-generated report |
| **Incremental Updates** | `src/summariser/incremental.py` | Assigns new articles to existing clusters via cosine similarity, recomputes centroids, regenerates report |
| **Chat (RAG)** | `src/summariser/urls/chat.py` | Embeds the user query, retrieves top articles via vector search, sends context + question to OpenAI |
| **Admin Panel** | `src/summariser/urls/admin.py` | UI for uploading JSONL source lists and adding URLs |
| **App / Routes** | `src/summariser/app.py` | FastAPI app, lifespan, report-viewer endpoint, latest-report API |

---

## Data Flow

### Full Report Generation (batch)

1. **Source list** — a JSONL file where each line is `{"link": "...", "title": "...", ...}`.
2. **Crawl** — Crawl4AI fetches each URL and converts the page to markdown. A pruning filter (threshold 0.7) strips boilerplate. HTTP URLs embedded in the text are removed.
3. **Embed** — Each article's markdown is encoded with `all-MiniLM-L6-v2` (384-dim, unit-normalized). Ingestion runs with 4 concurrent workers; each owns its own crawler and embedder instance.
4. **Persist** — The vector and metadata (url, file\_id, cluster\_id) are upserted into Qdrant. The markdown is written atomically to disk (temp file → rename). If either step fails the other is rolled back.
5. **Cluster** — All vectors are loaded, reduced to 10 dimensions with UMAP (cosine metric, 25 neighbors), then clustered with HDBSCAN (min cluster size 5). A centroid is chosen per cluster: the article whose 384-d vector is closest to the cluster mean.
6. **Summarise** — The centroid articles' markdown is concatenated and sent to OpenAI. The LLM produces a single cohesive report with topic headings and source citations.
7. **Save** — The report is written as `data/reports/YYYYMMDD_HHMM.md`.

### Incremental Update

When new URLs arrive after the initial report exists:

1. New articles are ingested normally (steps 2–4 above), each tagged `cluster_id = -1`.
2. Every new point's embedding is compared (dot product) against all stored centroids.
3. If the best similarity exceeds the threshold (default 0.45), the point is assigned to that centroid's cluster. Otherwise it stays as noise.
4. For every cluster that gained points, the centroid is recomputed from all cluster members. If the centroid article changes, its flag is updated in Qdrant.
5. A new report is generated using the updated centroids — no full re-clustering needed.

### Chat / RAG

1. The user's question is embedded with the same model used for articles.
2. A vector search retrieves the top matches (score ≥ 0.7) from Qdrant.
3. The latest compiled report and the retrieved article markdowns are assembled into a context window.
4. OpenAI answers the question grounded strictly in that context. Session continuity is maintained via `previous_response_id`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI + Uvicorn |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-d) |
| Vector database | Qdrant (local, file-based) |
| Clustering | UMAP + HDBSCAN |
| Web scraping | Crawl4AI |
| LLM | OpenAI API |
| Package manager | uv |

---

## Setup

1. **Prerequisites** — Python 3.10+, [uv](https://docs.astral.sh/uv/) installed.

2. **Install dependencies**
   ```sh
   uv sync
   ```

3. **Configure API key** — create a `.env` file in the repo root:
   ```
   OPENAI_API_KEY=sk-...
   ```

4. **Prepare a source list** — place a JSONL file in `data/`. Each line:
   ```json
   {"link": "https://example.com/article", "title": "Article title", "snippet": "..."}
   ```
   The project ships with `data/articles-500.jsonl` as a sample.

---

## Usage

### Run the web server

```sh
uv run uvicorn summariser.app:app --reload
```

Opens three UIs on `http://localhost:8000`:

| Path | Purpose |
|---|---|
| `/` | View the latest generated report |
| `/admin` | Upload a JSONL source list or paste URLs to trigger ingestion + report generation |
| `/chat` | Ask questions about the ingested articles |

### CLI scripts

**Ingest all URLs from the default JSONL source list:**
```sh
uv run python scripts/inject_urls.py
```

**Generate a report from already-ingested articles:**
```sh
uv run python scripts/create_report.py
```

**Incremental update — add new URLs and regenerate the report:**
```sh
uv run python scripts/update_report_with_new_urls.py --urls https://example.com/a https://example.com/b

# From a file (one URL per line, # for comments):
uv run python scripts/update_report_with_new_urls.py --urls-file new_sources.txt

# Bootstrap: run full clustering once before switching to incremental mode:
uv run python scripts/update_report_with_new_urls.py --urls https://example.com --bootstrap
```

---

## Configuration

All tunables live in `src/summariser/config.py` and can be overridden via environment variables:

| Key | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. Set in `.env`. |
| `CENTROID_ASSIGN_MIN_COSINE` | `0.45` | Minimum cosine similarity for a new article to be assigned to an existing cluster during incremental updates. |
| `DB_PATH` | `qdrant_storage/` | Local Qdrant storage directory |
| `COLLECTION_NAME` | `article_embeddings_500_2` | Qdrant collection name |
| `URLS_PATH` | `data/articles-500.jsonl` | Default source list for batch ingestion |
| `MARKDOWN_PATH` | `data/markdown-500/` | Where crawled article markdowns are stored |
| `REPORTS_PATH` | `data/reports/` | Where generated reports are saved |

---

## Known Limitations / TODOs

- **Embedding truncation** — `all-MiniLM-L6-v2` has a hard 256-token limit. Longer articles are silently truncated before embedding.
- **Crawl quality** — no post-crawl validation on markdown quality; low-content pages may produce poor embeddings.
- **Cluster scaling** — UMAP + HDBSCAN performance and quality may degrade at very large scale (tens of thousands of articles). Consider approximate nearest-neighbor approaches or periodic full re-clustering.
- **Article quality filtering** — articles with very little content should be disqualified or merged with related articles before summarisation.
- **Report length** — long reports could be broken into multiple focused articles for readability.
- **Batch embeddings** — articles are currently embedded one at a time per worker; batched inference would improve throughput.
