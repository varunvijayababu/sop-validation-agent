from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

logger.info(
    "Loading embedding model: all-MiniLM-L6-v2"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

logger.info(
    "Embedding model loaded successfully"
)


def create_embedding(text):

    try:

        logger.info(
            f"Generating embedding. Text length: {len(text)}"
        )

        embedding = model.encode(text).tolist()

        logger.info(
            f"Embedding generated successfully. Vector size: {len(embedding)}"
        )

        return embedding

    except Exception as e:

        logger.exception(
            f"Embedding generation failed: {str(e)}"
        )

        raise