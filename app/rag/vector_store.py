from app.rag.section_ranker import (
    rank_sections
)

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
            "Generating section weights"
        )

        section_weights = rank_sections(
            chunks
        )

        weight_map = {
            item["section"]: item["weight"]
            for item in section_weights
        }

        for chunk in chunks:

            if chunk["section"] not in weight_map:
                raise Exception(
                    f"Weight missing for section: {chunk['section']}"
                )

            logger.info(
                f"Chunk {i+1}: "
                f"{chunk['section']} "
                f"(Page {chunk['page']}) "
                f"| Weight={chunk['weight']}%"
            )
        
        total_weight = sum(
            chunk["weight"]
            for chunk in chunks
        )

        logger.info(
            f"FINAL STORED WEIGHT TOTAL: {total_weight}"
        )

        logger.info(
            f"Total chunks generated: {len(chunks)}"
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
                        "page": chunk["page"],
                        "weight": chunk["weight"]
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