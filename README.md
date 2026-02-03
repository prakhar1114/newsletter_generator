## summariser

run app
```
uv run uvicorn summariser.app:app --reload
```

generate a compiled report
```
uv run python -c "from summariser.reporting import generate_compiled_report; md, path = generate_compiled_report(); print(path)"
```


Notes:
- using file based qdrant client instead of a server
- using MiniLML6-v2 as the embedding model: this has a hard 256 token-limit and other things are truncated



TODO:
- batch embeddings for performance
- verify markdowns from urls - crawl4ai
- umap and hdbscan vs dbscan- curse of dimensionality at scale
- define data richness and disqualify low quality articles (to alwys have good summaries)+ club multiple articles if the token count is less for more details
- break it down into multiple articles
