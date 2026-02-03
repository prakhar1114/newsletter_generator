from __future__ import annotations

from typing import Any, Iterable

from .config import COLLECTION_NAME, DB_PATH
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)


client: QdrantClient = QdrantClient(path=DB_PATH)

def init_vector_db(collection_name: str = COLLECTION_NAME) -> None:
    """
    Ensure the configured collection exists.
    """
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )


def save_to_db(point_id: int | str, vector: list[float], metadata: dict[str, Any]) -> None:
    """
    Stores an embedding and its associated metadata.
    """
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload=metadata
            )
        ]
    )
    print(f"Point {point_id} saved successfully.")


def delete_from_db(point_id: int | str) -> None:
    """
    Deletes a point by id. Useful for rollback when using local Qdrant storage.
    """
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=PointIdsList(points=[point_id]),
    )


def set_payload_for_points(
    point_ids: Iterable[int | str],
    payload: dict[str, Any],
    *,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """
    Merge `payload` into existing payload for each point id.
    """
    client.set_payload(
        collection_name=collection_name,
        payload=payload,
        points=list(point_ids),
    )


def retrieve_points(
    point_ids: Iterable[int | str],
    *,
    collection_name: str = COLLECTION_NAME,
    with_vectors: bool = True,
    with_payload: bool = True,
):
    """
    Retrieve points by id.
    """
    return client.retrieve(
        collection_name=collection_name,
        ids=list(point_ids),
        with_vectors=with_vectors,
        with_payload=with_payload,
    )


def scroll_points(
    *,
    collection_name: str = COLLECTION_NAME,
    scroll_filter: Filter | None = None,
    limit: int = 1000,
    with_vectors: bool = True,
    with_payload: bool = True,
):
    """
    Scroll points (single call). Returns list of points.
    """
    init_vector_db(collection_name)
    points, _next_page_offset = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=limit,
        with_vectors=with_vectors,
        with_payload=with_payload,
    )
    return points


def centroid_filter() -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="is_centroid",
                match=MatchValue(value=True),
            )
        ]
    )


def cluster_filter(cluster_id: int) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="cluster_id",
                match=MatchValue(value=int(cluster_id)),
            )
        ]
    )


def centroid_points(
    *,
    collection_name: str = COLLECTION_NAME,
    limit: int = 1000,
    with_vectors: bool = True,
    with_payload: bool = True,
):
    """
    Return points where payload `is_centroid == True`.
    """
    return scroll_points(
        collection_name=collection_name,
        scroll_filter=centroid_filter(),
        limit=limit,
        with_vectors=with_vectors,
        with_payload=with_payload,
    )


def points_in_cluster(
    cluster_id: int,
    *,
    collection_name: str = COLLECTION_NAME,
    limit: int = 2000,
    with_vectors: bool = True,
    with_payload: bool = True,
):
    """
    Return points where payload `cluster_id == <cluster_id>`.
    """
    return scroll_points(
        collection_name=collection_name,
        scroll_filter=cluster_filter(cluster_id),
        limit=limit,
        with_vectors=with_vectors,
        with_payload=with_payload,
    )


def url_exists(url: str) -> bool:
    """
    Returns True if we already have a point with payload `url == <url>`.
    """
    # Ensure collection exists so scroll doesn't error on first run.
    init_vector_db()
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="url",
                    match=MatchValue(value=url),
                )
            ]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(points) > 0


def search_with_filter(query_vector: list[float], limit: int = 20):
    # Using 'query_points' instead of 'search'
    # This is the preferred method in the latest qdrant-client versions
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,

        limit=limit
    ).points
