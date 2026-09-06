import logging
from typing import Sequence

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384


class EmbeddingService:
    """
    Embedding service using sentence-transformers (all-MiniLM-L6-v2).
    Generates 384-dimensional dense vector embeddings for text chunks and queries.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Returns the embedding dimension for this model (384)."""
        model = self._get_model()
        return model.get_embedding_dimension()

    def encode(self, text: str) -> list[float]:
        """Encode a single text into a normalized embedding vector."""
        if not text:
            return [0.0] * DEFAULT_EMBEDDING_DIMENSION
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()

    def encode_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a batch of texts into normalized embedding vectors."""
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]


# Singleton instance for shared usage
embedding_service = EmbeddingService()
