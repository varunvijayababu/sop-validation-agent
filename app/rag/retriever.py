from app.rag.embedder import create_embedding
from app.rag.qdrant_client import client

import logging

logger = logging.getLogger(__name__)

COLLECTION_NAME = "sop_reference_docs"


def retrieve_context(query):

    try:

        logger.info(
            "Starting context retrieval"
        )

        logger.info(
            f"Query length: {len(query)} characters"
        )

        query_vector = create_embedding(query)

        logger.info(
            "Query embedding generated"
        )

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=15
        )

        logger.info(
            f"Retrieved {len(results.points)} results from Qdrant"
        )

        retrieved_sections = []

        for idx, point in enumerate(results.points):

            section = point.payload["section"]
            page = point.payload["page"]

            logger.info(
                f"Result {idx + 1}: "
                f"{section} "
                f"(Page {page})"
            )

            retrieved_sections.append(
                {
                    "section": section,
                    "page": page,
                    "text": point.payload["text"]
                }
            )

        logger.info(
            "Context retrieval completed successfully"
        )

        return retrieved_sections

    except Exception as e:

        logger.exception(
            f"Context retrieval failed: {str(e)}"
        )

        raise