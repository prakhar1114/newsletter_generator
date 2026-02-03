import os
from .config import DB_PATH, COLLECTION_NAME
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PointIdsList,
)


client: QdrantClient = QdrantClient(path=DB_PATH)

def init_vector_db():
    # Ensure collection exists
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )


def save_to_db(point_id: int | str, vector: list[float], metadata: dict):
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

def search_with_filter(query_vector: list, limit: int = 20):
    # Using 'query_points' instead of 'search'
    # This is the preferred method in the latest qdrant-client versions
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,

        limit=limit
    ).points
