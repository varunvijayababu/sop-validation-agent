from qdrant_client import QdrantClient
import logging

logger = logging.getLogger(__name__)

try:

    logger.info(
        "Connecting to local Qdrant..."
    )

    client = QdrantClient(
        host="localhost",
        port=6333
    )

    collections = client.get_collections()

    logger.info(
        f"Connected to Qdrant successfully. "
        f"Collections found: {len(collections.collections)}"
    )

except Exception as e:

    logger.exception(
        f"Failed to connect to Qdrant: {str(e)}"
    )

    raise