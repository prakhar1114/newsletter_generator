import os
from pathlib import Path
DB_PATH = "/Users/prakharjain/code/summariser/qdrant_storage"
# COLLECTION_NAME = "article_embeddings"


# URLS_PATH = Path("/Users/prakharjain/code/summariser/data/articles-150.jsonl")
# URLS_PATH = Path("/Users/prakharjain/code/summariser/data/test_articles.jsonl")
# MARKDOWN_PATH = Path("/Users/prakharjain/code/summariser/data/markdowns")
REPORTS_PATH = Path("/Users/prakharjain/code/summariser/data/reports")


COLLECTION_NAME = "article_embeddings_500"
URLS_PATH = Path("/Users/prakharjain/code/summariser/data/articles-500.jsonl")
MARKDOWN_PATH = Path("/Users/prakharjain/code/summariser/data/markdown-500")

# Incremental clustering/reporting knobs
CENTROID_ASSIGN_MIN_COSINE = float(os.getenv("CENTROID_ASSIGN_MIN_COSINE", "0.45"))
