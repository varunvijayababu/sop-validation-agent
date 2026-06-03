from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)
from app.rag.embedder import create_embedding
from app.rag.chunker import chunk_text
from app.rag.qdrant_client import client

COLLECTION_NAME = "sop_reference_docs"


def create_collection():

    collections = client.get_collections()

    existing = [
        c.name
        for c in collections.collections
    ]

    if COLLECTION_NAME not in existing:

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )
        
def store_document(text):

    create_collection()

    chunks = chunk_text(text)

    points = []

    for idx, chunk in enumerate(chunks):

        embedding = create_embedding(chunk)

        points.append(
            PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "text": chunk
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )