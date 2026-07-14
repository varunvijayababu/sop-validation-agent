import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

try:
    if qdrant_url:
        logger.info(f"Connecting to Qdrant Cloud at {qdrant_url}...")
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
    else:
        logger.info("Connecting to local Qdrant...")
        client = QdrantClient(
            host="localhost",
            port=6333
        )
except Exception as e:
    logger.exception(f"Failed to initialize Qdrant client: {str(e)}")
    raise

def verify_connection() -> bool:
    """Verifies that the Qdrant connection is active by fetching collections."""
    try:
        collections = client.get_collections()
        logger.info(
            f"Connected to Qdrant successfully. "
            f"Collections found: {len(collections.collections)}"
        )
        return True
    except Exception as e:
        logger.exception(f"Failed to connect to Qdrant: {str(e)}")
        return False