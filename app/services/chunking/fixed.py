"""Fixed-size chunker with configurable window length and overlap."""

from app.services.chunking.base import Chunker


class FixedChunker(Chunker):
    """Chunks text into windows of fixed character length with configurable overlap.

    Respects word boundaries when possible to avoid mid-word truncation.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def chunk(self, text: str) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return []

        if len(cleaned) <= self.chunk_size:
            return [cleaned]

        chunks: list[str] = []
        start = 0
        text_len = len(cleaned)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            if end < text_len:
                # Search backwards for whitespace within the last 20% of the chunk window
                lookback_limit = max(start, end - int(self.chunk_size * 0.2))
                last_space = -1
                for i in range(end - 1, lookback_limit - 1, -1):
                    if cleaned[i].isspace():
                        last_space = i
                        break

                if last_space > start:
                    end = last_space

            chunk_content = cleaned[start:end].strip()
            if chunk_content:
                chunks.append(chunk_content)

            # If we reached the end of the text, we are done
            if end >= text_len:
                break

            # Advance start by (end - overlap)
            step = max(1, (end - start) - self.chunk_overlap)
            start += step

        return chunks
