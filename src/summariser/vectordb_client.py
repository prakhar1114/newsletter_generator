import os
from .config import DB_PATH, COLLECTION_NAME
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue


client: QdrantClient = QdrantClient(path=DB_PATH)

def init_vector_db():
    # Ensure collection exists
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )


def save_to_db(point_id: int, vector: list, metadata: dict):
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

def search_with_filter(query_vector: list, category_filter: str, limit: int = 5):
    # Using 'query_points' instead of 'search'
    # This is the preferred method in the latest qdrant-client versions
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="category",
                    match=MatchValue(value=category_filter)
                )
            ]
        ),
        limit=limit
    ).points
