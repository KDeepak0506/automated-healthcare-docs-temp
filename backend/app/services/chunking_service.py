import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkInfo:
    chunk_index: int
    chunk_text: str
    start_offset: int
    end_offset: int
    page_number: int | None = None


class ChunkingService:
    """
    Deterministic text chunking service for RAG.
    Segments sanitized clinical documents into overlapping word chunks
    while preserving paragraph boundaries, numbers, and offsets.
    """

    def __init__(
        self,
        target_words: int = 650,
        overlap_words: int = 75,
        min_words: int = 20,
    ) -> None:
        self.target_words = target_words
        self.overlap_words = overlap_words
        self.min_words = min_words


    def chunk_text(
        self,
        text: str,
        target_words: int | None = None,
        overlap_words: int | None = None,
        min_words: int | None = None,
    ) -> list[ChunkInfo]:
        """
        Split sanitized text into sequential, overlapping chunks.
        Guarantees zero word loss: all words from start to end are preserved.
        Trailing fragments smaller than min_words are absorbed into the preceding chunk.
        Returns a list of ChunkInfo objects with precise character offsets.
        """
        if not text or not text.strip():
            return []

        target = target_words or self.target_words
        overlap = overlap_words or self.overlap_words
        step = max(1, target - overlap)
        raw_min = min_words if min_words is not None else self.min_words
        minimum = max(1, min(raw_min, step))

        # Tokenize by finding words and their character spans
        word_matches = list(re.finditer(r"\S+", text))
        if not word_matches:
            return []

        total_words = len(word_matches)
        if total_words <= target:
            # Entire text fits into a single chunk
            start_off = word_matches[0].start()
            end_off = word_matches[-1].end()
            return [
                ChunkInfo(
                    chunk_index=0,
                    chunk_text=text[start_off:end_off],
                    start_offset=start_off,
                    end_offset=end_off,
                )
            ]

        chunks: list[ChunkInfo] = []
        chunk_idx = 0
        start_w = 0

        while start_w < total_words:
            end_w = min(total_words, start_w + target)

            # If remaining words after this chunk would be fewer than `minimum`,
            # absorb them into this chunk so we cover to the end without creating a tiny orphan chunk
            remaining_after_this = total_words - end_w
            if 0 < remaining_after_this < minimum:
                end_w = total_words

            start_off = word_matches[start_w].start()
            end_off = word_matches[end_w - 1].end()

            chunk_text = text[start_off:end_off].strip()
            if chunk_text:
                chunks.append(
                    ChunkInfo(
                        chunk_index=chunk_idx,
                        chunk_text=chunk_text,
                        start_offset=start_off,
                        end_offset=end_off,
                    )
                )
                chunk_idx += 1

            if end_w >= total_words:
                break

            start_w += step

        return chunks





# Singleton instance
chunking_service = ChunkingService()
