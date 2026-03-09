"""RAG search using sentence embeddings."""

import logging
from typing import Any, Callable, List, Optional

from ollama import Client

from .config import Config
from .models import SearchResult, CitationReturnMode
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class SearchEngine:
    """Sentence-focused RAG search."""

    def __init__(self, config: Config, vector_store: VectorStore | None = None):
        self.config = config
        self.vector_store = vector_store or VectorStore(str(config.VECTOR_STORE_DIR))

        if (
            self.config.EMBEDDING_DIMENSIONS > 0
            and self.vector_store.has_dimension_mismatch(
                self.config.EMBEDDING_DIMENSIONS
            )
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

    def _get_store_dimension(self) -> int | None:
        detected = self.vector_store.get_detected_dimension()
        return detected if isinstance(detected, int) and detected > 0 else None

    def _validate_query_embedding(self, embedding: List[float]) -> List[float]:
        actual_dim = len(embedding or [])
        if actual_dim <= 0:
            raise ValueError("Embedding provider returned an empty query embedding")

        expected_dim = int(getattr(self.config, "EMBEDDING_DIMENSIONS", 0) or 0)
        if expected_dim > 0 and actual_dim != expected_dim:
            raise ValueError(
                "Embedding provider returned "
                f"a {actual_dim}-dimensional query vector but EMBEDDING_DIMENSIONS={expected_dim}. "
                "This usually means the model/server ignored the requested dimensions."
            )

        store_dim = self._get_store_dimension()
        if store_dim and actual_dim != store_dim:
            raise ValueError(
                "Embedding provider returned "
                f"a {actual_dim}-dimensional query vector but the vector store uses {store_dim}. "
                "Clear the vector store or re-embed with a consistent model/dimension setting."
            )

        return [float(x) for x in embedding]

    def _get_query_embedding(self, query: str) -> List[float]:
        client = Client(host=self.config.OLLAMA_BASE_URL)
        response = client.embeddings(
            model=self.config.EMBEDDING_MODEL,
            prompt=query,
            options=self._get_embedding_options(),
        )
        return self._validate_query_embedding(response["embedding"])

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
        return self.search_best_sentences(
            query=query, document_key=document_key, top_sentences=top_k
        )

    def search_best_sentences(
        self,
        query: str,
        document_key: Optional[str] = None,
        top_sections: int = 5,
        top_sentences: int = 10,
        ensure_sentence_embeddings: bool = True,
        citation_return_mode: CitationReturnMode = "sentence",
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> List[SearchResult]:
        """Return the best matching embedded sentences.

        This method is designed to scale to very large sentence collections.
        It relies on the persistent vector index to do the similarity search and
        only pulls back the top-k ids + distances, then fetches just those texts.

        Args:
            query: Query string
            document_key: If provided, restrict to this document.
            top_sections: currently unused in sentence-only mode.
            top_sentences: we return up to top_sentences sentences.
            ensure_sentence_embeddings: If True, ensures that sentence embeddings are available.
            citation_return_mode:
                - "sentence": return sentence only (current behavior)
                - "bibtex": return only the referenced BibTeX entries for the matched sentence
                - "both": return the sentence plus referenced BibTeX entries
        """
        if ensure_sentence_embeddings and not self.config.AUTO_EMBED_SENTENCES:
            # No auto-embedding here; embedding is handled by EmbeddingManager.
            pass

        def emit_progress(payload: dict[str, Any]) -> None:
            if not progress_callback:
                return
            try:
                progress_callback(payload)
            except Exception:
                logger.debug("Search progress callback failed", exc_info=True)

        # Keep signature compatibility; top_sections is currently unused in sentence-only mode.
        _ = top_sections

        logger.debug("Embedding query: %s", query)

        emit_progress(
            {
                "stage": "embedding",
                "percentage": 20,
                "message": "Embedding sentence",
                "detail": "Generating the query embedding",
            }
        )

        query_embedding = self._get_query_embedding(query)

        logger.debug("Searching query: %s", query_embedding)

        emit_progress(
            {
                "stage": "semantic_search",
                "percentage": 30,
                "message": "Searching for similar sentences",
                "detail": "Query embedding generated, now searching the vector index for similar sentences",
            }
        )

        # Ask the vector DB for the best matches; do NOT ask for embeddings.
        ids, distances, metadatas = self.vector_store.search_sentence_ids(
            query_embedding=query_embedding,
            document_key=document_key,
            top_k=max(1, int(top_sentences)),
        )

        if not ids:
            emit_progress(
                {
                    "stage": "vector_search",
                    "percentage": 45,
                    "message": "Found 0 similar sentences",
                    "detail": "No semantic matches were returned",
                    "similar_sentences": 0,
                }
            )
            return []

        id_to_text = self.vector_store.get_sentence_texts_by_ids(ids)

        emit_progress(
            {
                "stage": "metadata_fetch",
                "percentage": 45,
                "message": f"Found {len(ids)} similar sentence{'s' if len(ids) != 1 else ''}. Fetching metadata",
                "detail": "Fetched similar sentence texts, now processing results",
                "similar_sentences": 0,
            }
        )

        results: list[SearchResult] = []
        for i, sid in enumerate(ids):
            text = id_to_text.get(sid, "")
            emit_progress(
                {
                    "stage": "metadata_fetch",
                    "percentage": 45 + int((i + 1) / len(ids) * 15),
                    "message": f"Processing sentence {i + 1} of {len(ids)}",
                    "detail": "Processing similar sentences and fetching metadata for sentence.",
                    "similar_sentences": 0,
                }
            )
            if not text:
                continue

            meta = (metadatas[i] if i < len(metadatas) else None) or {}
            dk = str(meta.get("document_key", ""))

            score = distances[i]

            cited_bibtex = list(meta.get("referenced_bibtex") or [])
            cited_texts = list(meta.get("referenced_texts") or [])
            citation_numbers = list(meta.get("citation_numbers") or [])

            bibtex_block = "\n\n".join([b for b in cited_bibtex if b])
            if citation_return_mode == "bibtex":
                shaped_text = bibtex_block
            elif citation_return_mode == "both":
                shaped_text = text
                if bibtex_block:
                    shaped_text = f"{text}\n\n{bibtex_block}"
            else:
                shaped_text = text

            results.append(
                SearchResult(
                    text=shaped_text,
                    document_title="",
                    section_title="",
                    zotero_key=dk,
                    relevance_score=score,
                    rerank_score=score,
                    cited_bibtex=cited_bibtex,
                    cited_reference_texts=cited_texts,
                    citation_numbers=citation_numbers,
                )
            )

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        results = results[: max(1, int(top_sentences))]
        emit_progress(
            {
                "stage": "vector_search",
                "percentage": 60,
                "message": f"Found {len(results)} similar sentence{'s' if len(results) != 1 else ''}",
                "detail": "Semantic search finished",
                "similar_sentences": len(results),
            }
        )
        return results

    def get_stats(self) -> dict:
        return {
            "total_sentences": self.vector_store.get_sentence_count(),
            "embedded_documents": len(self.vector_store.get_embedded_documents()),
        }
