from app.rag.embedder import create_embedding
from app.rag.qdrant_client import client

COLLECTION_NAME = "sop_reference_docs"

def retrieve_context(query):

    query_vector = create_embedding(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5
    )

    context = []

    for point in results.points:
        context.append(
            point.payload["text"]
        )

    return "\n".join(context)