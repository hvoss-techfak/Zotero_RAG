"""RAG search using sentence embeddings."""

import logging
from typing import List, Optional

import ollama

from .config import Config
from .models import SearchResult
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class SearchEngine:
    """Sentence-focused RAG search."""

    def __init__(self, config: Config):
        self.config = config
        self.vector_store = VectorStore(str(config.VECTOR_STORE_DIR))

        if self.config.EMBEDDING_DIMENSIONS > 0 and self.vector_store.has_dimension_mismatch(
            self.config.EMBEDDING_DIMENSIONS
        ):
            detected = self.vector_store.get_detected_dimension()
            logger.warning(
                "Dimension mismatch: config expects %s but existing embeddings are %s. "
                "Set EMBEDDING_DIMENSIONS=%s or re-embed all documents.",
                self.config.EMBEDDING_DIMENSIONS,
                detected,
                detected,
            )

    def _get_embedding_options(self) -> dict:
        opts = {}
        if self.config.EMBEDDING_DIMENSIONS > 0:
            opts["dimensions"] = self.config.EMBEDDING_DIMENSIONS
        return opts

    def _get_query_embedding(self, query: str) -> List[float]:
        response = ollama.embeddings(
            model=self.config.EMBEDDING_MODEL,
            prompt=query,
            options=self._get_embedding_options(),
        )
        return response["embedding"]

    def search(
        self,
        query: str,
        document_key: Optional[str] = None,
        top_sentences: int = 3,
    ) -> List[SearchResult]:
        """Search embedded sentences.

        Args:
            query: Query string
            document_key: If provided, restrict to this document.
            top_sentences: we return up to top_sentences sentences.
        """
        top_k = max(1, top_sentences)
        return self.search_best_sentences(query=query, document_key=document_key, top_sentences=top_k)

    def search_best_sentences(
        self,
        query: str,
        document_key: Optional[str] = None,
        top_sections: int = 5,
        top_sentences: int = 10,
        ensure_sentence_embeddings: bool = True,
    ) -> List[SearchResult]:
        """Return the best matching embedded sentences.

        This method is designed to scale to very large sentence collections.
        It relies on the persistent vector index to do the similarity search and
        only pulls back the top-k ids + distances, then fetches just those texts.
        """
        if ensure_sentence_embeddings and not self.config.AUTO_EMBED_SENTENCES:
            # No auto-embedding here; embedding is handled by EmbeddingManager.
            pass

        # Keep signature compatibility; top_sections is currently unused in sentence-only mode.
        _ = top_sections

        query_embedding = self._get_query_embedding(query)

        # Ask the vector DB for the best matches; do NOT ask for embeddings.
        ids, distances, metadatas = self.vector_store.search_sentence_ids(
            query_embedding=query_embedding,
            document_key=document_key,
            top_k=max(1, int(top_sentences)),
        )

        if not ids:
            return []

        id_to_text = self.vector_store.get_sentence_texts_by_ids(ids)

        results: list[SearchResult] = []
        for i, sid in enumerate(ids):
            text = id_to_text.get(sid, "")
            if not text:
                continue

            meta = (metadatas[i] if i < len(metadatas) else None) or {}
            dk = str(meta.get("document_key", ""))

            # Chroma returns a distance, where smaller == better. Convert to a monotonic
            # score where larger == better to fit existing SearchResult sorting.
            dist = float(distances[i]) if i < len(distances) else 0.0
            score = 1.0 / (1.0 + max(0.0, dist))

            results.append(
                SearchResult(
                    text=text,
                    document_title="",
                    section_title="",
                    zotero_key=dk,
                    relevance_score=score,
                    rerank_score=score,
                )
            )

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[: max(1, int(top_sentences))]

    def get_stats(self) -> dict:
        return {
            "total_sentences": self.vector_store.get_sentence_count(),
            "embedded_documents": len(self.vector_store.get_embedded_documents()),
        }
