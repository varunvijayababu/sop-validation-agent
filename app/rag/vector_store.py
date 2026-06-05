from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

from app.rag.embedder import create_embedding
from app.rag.qdrant_client import client
from app.rag.text_splitter import split_sections

import logging

logger = logging.getLogger(__name__)

COLLECTION_NAME = "sop_reference_docs"


def create_collection():

    logger.info(
        "Checking if Qdrant collection exists"
    )

    collections = client.get_collections()

    existing = [
        c.name
        for c in collections.collections
    ]

    if COLLECTION_NAME not in existing:

        logger.info(
            f"Creating collection: {COLLECTION_NAME}"
        )

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )

        logger.info(
            f"Collection created: {COLLECTION_NAME}"
        )

    else:

        logger.info(
            f"Collection already exists: {COLLECTION_NAME}"
        )


def store_document(pages):

    try:

        logger.info(
            "Starting document storage process"
        )

        create_collection()

        chunks = split_sections(pages)

        logger.info(
            f"Total chunks generated: {len(chunks)}"
        )

        for i, chunk in enumerate(chunks):

            logger.info(
                f"Chunk {i + 1}: "
                f"{chunk['section']} "
                f"(Page {chunk['page']})"
            )

        points = []

        for idx, chunk in enumerate(chunks):

            logger.info(
                f"Generating embedding for chunk {idx + 1}"
            )

            embedding = create_embedding(
                chunk["text"]
            )

            points.append(
                PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        "section": chunk["section"],
                        "page": chunk["page"]
                    }
                )
            )

        logger.info(
            f"Uploading {len(points)} vectors to Qdrant"
        )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        logger.info(
            "Document stored successfully in Qdrant"
        )

    except Exception as e:

        logger.exception(
            f"Vector store failed: {str(e)}"
        )

        raise