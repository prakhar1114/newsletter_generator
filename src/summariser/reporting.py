from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import hdbscan
import numpy as np
import umap
from openai import OpenAI

from summariser.config import COLLECTION_NAME, REPORTS_PATH
from summariser.utils import fetch_markdown_from_id
from summariser.vectordb_client import client, init_vector_db

logger = logging.getLogger(__name__)

# Ensure logs show up in scripts/notebooks by default, without overriding
# an app's logging configuration if it already exists.
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass(frozen=True)
class ClusterCentroid:
    cluster_id: int
    index: int  # index into the vectors/payload arrays
    file_id: str
    url: str


def _scroll_all_points(
    *,
    collection_name: str,
    limit: int = 1000,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Fetch points from Qdrant Local using a single scroll().

    Returns:
      - vectors: (N, 384) float32 array
      - payloads: list of payload dicts (must contain url + file_id)
    """
    init_vector_db()

    # For now we intentionally do a single scroll call (like the notebook),
    # which avoids offset/loop edge-cases in local mode.
    points, _next_page_offset = client.scroll(
        collection_name=collection_name,
        with_vectors=True,
        with_payload=True,
        limit=limit,
    )

    if not points:
        return np.empty((0, 0), dtype=np.float32), []

    vectors = np.stack([np.asarray(p.vector, dtype=np.float32) for p in points], axis=0)
    payloads: list[dict[str, Any]] = [p.payload or {} for p in points]
    return vectors, payloads


def _cluster_labels(
    vectors: np.ndarray,
    *,
    umap_neighbors: int = 25,
    umap_components: int = 10,
    hdbscan_min_cluster_size: int = 5,
) -> np.ndarray:
    """
    UMAP reduce then HDBSCAN cluster. Returns labels of length N.
    """
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        return np.array([], dtype=np.int32)

    reducer = umap.UMAP(
        n_neighbors=umap_neighbors,
        n_components=umap_components,
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(vectors)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        metric="euclidean",
        prediction_data=False,
    )
    labels = clusterer.fit_predict(reduced)
    return labels.astype(np.int32)


def _pick_centroid_article_per_cluster(
    vectors: np.ndarray,
    payloads: list[dict[str, Any]],
    labels: np.ndarray,
) -> list[ClusterCentroid]:
    """
    For each cluster, compute centroid in original embedding space and pick the
    closest article (centroid article).
    """
    if len(payloads) != vectors.shape[0]:
        raise ValueError("payloads length must match vectors rows")

    centroids: list[ClusterCentroid] = []
    for label in sorted(set(int(x) for x in labels.tolist())):
        if label == -1:
            continue  # noise
        idxs = np.where(labels == label)[0]
        if idxs.size == 0:
            continue
        cluster_vectors = vectors[idxs]
        centroid = cluster_vectors.mean(axis=0)

        # Choose the article closest to centroid (L2 distance).
        dists = np.linalg.norm(cluster_vectors - centroid, axis=1)
        best_local = int(np.argmin(dists))
        best_index = int(idxs[best_local])

        payload = payloads[best_index] or {}
        file_id = str(payload.get("file_id", "")).strip()
        url = str(payload.get("url", "")).strip()
        if not file_id:
            # If payload is incomplete, skip this cluster.
            continue
        centroids.append(
            ClusterCentroid(
                cluster_id=int(label),
                index=best_index,
                file_id=file_id,
                url=url,
            )
        )
    return centroids


def _build_centroid_context(centroids: Iterable[ClusterCentroid]) -> list[str]:
    """
    Loads markdown for each centroid article and formats it for the LLM.
    """
    blocks: list[str] = []
    for c in centroids:
        md = fetch_markdown_from_id(c.file_id)
        blocks.append(
            "\n".join(
                [
                    # f"cluster_id: {c.cluster_id}",
                    # f"file_id: {c.file_id}",
                    f"source: {c.url}",
                    md,
                ]
            )
        )
    return blocks


def generate_compiled_report(
    *,
    model: str = "gpt-5-mini",
    system_prompt: str | None = None,
    collection_name: str = COLLECTION_NAME,
    limit: int = 1000,
    openai_client: OpenAI | None = None,
) -> tuple[str, Path]:
    """
    Generates a single compiled markdown report and writes it to:
      REPORTS_PATH/<YYYYMMDD_HHMMSS>.md

    Uses only the centroid article per cluster as context.
    """
    logger.info("[report] loading points from collection=%r ...", collection_name)
    vectors, payloads = _scroll_all_points(
        collection_name=collection_name,
        limit=limit,
    )
    if vectors.size == 0:
        raise RuntimeError("No vectors found in Qdrant collection")
    logger.info("[report] loaded %s vectors", vectors.shape[0])

    logger.info("[report] clustering (UMAP → HDBSCAN) ...")
    labels = _cluster_labels(vectors)
    centroids = _pick_centroid_article_per_cluster(vectors, payloads, labels)
    if not centroids:
        raise RuntimeError("No clusters found (all noise or insufficient points)")
    logger.info("[report] clusters=%s (centroid articles selected)", len(centroids))

    logger.info("[report] loading centroid markdown from disk ...")
    context_blocks = _build_centroid_context(centroids)
    combined_content = "\n\n---\n\n".join(context_blocks)

    if system_prompt is None:
        system_prompt = (
            "You are a professional research analyst. Summarize the provided articles into a single, "
            "cohesive topic report formatted as markdown. Identify the core themes, key events, and "
            "common entities across clusters. Always cite sources in the report. Just return the report, no other text."
        )

    logger.info("[report] calling OpenAI model=%r ...", model)
    oai = openai_client or OpenAI()
    resp = oai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Articles:\n\n{combined_content}"},
        ],
    )
    report_md = resp.choices[0].message.content or ""

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out_path = REPORTS_PATH / f"{ts}.md"
    out_path.write_text(report_md, encoding="utf-8")
    logger.info("[report] wrote %s", out_path)

    return report_md, out_path
