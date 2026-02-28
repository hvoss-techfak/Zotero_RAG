"""Two-stage RAG search with reranking using Ollama."""

import logging
from typing import List, Optional
import ollama

from .config import Config
from .models import SearchResult, Document, SentenceWindow
from .embedding_manager import EmbeddingManager
from .vector_store import VectorStore


logger = logging.getLogger(__name__)


class SearchEngine:
    """Two-stage RAG search: section search then sentence reranking."""

    def __init__(self, config: Config):
        self.config = config
        self.vector_store = VectorStore(config.VECTOR_STORE_DIR)
        # EmbeddingManager for reranking (lazy init)
        self._embedding_manager: Optional[EmbeddingManager] = None
        
        # Check for dimension mismatch on init and warn user
        if self.config.EMBEDDING_DIMENSIONS > 0:
            if self.vector_store.has_dimension_mismatch(self.config.EMBEDDING_DIMENSIONS):
                detected = self.vector_store.get_detected_dimension()
                logger.warning(
                    f"Dimension mismatch: config expects {self.config.EMBEDDING_DIMENSIONS} "
                    f"but existing embeddings are {detected} dimensions. "
                    f"Set EMBEDDING_DIMENSIONS={detected} or re-embed all documents."
                )

    @property
    def embedding_manager(self) -> EmbeddingManager:
        """Lazy initialization of embedding manager for reranking."""
        if self._embedding_manager is None:
            self._embedding_manager = EmbeddingManager(self.config)
        return self._embedding_manager

    # --- Stage 1: Section search ---

    def _get_embedding_options(self) -> dict:
        """Get Ollama options for embedding, using config dimensions if set."""
        opts = {}
        if self.config.EMBEDDING_DIMENSIONS > 0:
            opts["dimensions"] = self.config.EMBEDDING_DIMENSIONS
        return opts

    def _get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for the query."""
        response = ollama.embeddings(
            model=self.config.EMBEDDING_MODEL,
            prompt=query,
            options=self._get_embedding_options()
        )
        return response["embedding"]

    def search_sections(
        self,
        query: str,
        top_k: int = 10
    ) -> List[tuple[str, float]]:
        """Find relevant sections using embedding similarity."""
        query_embedding = self._get_query_embedding(query)
        section_ids, embeddings = self.vector_store.search_sections(
            query_embedding, top_k=top_k
        )

        # Handle empty results (can happen with dimension mismatches)
        if not section_ids or not embeddings:
            return []

        # Get sections and compute cosine similarity scores
        results = []
        for i, section_id in enumerate(section_ids):
            if i >= len(embeddings) or embeddings[i] is None:
                continue
            section = self.vector_store.get_section(section_id)
            if section:
                score = self._cosine_similarity(query_embedding, embeddings[i])
                results.append((section_id, score))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def _cosine_similarity(
        self,
        a: List[float],
        b: List[float]
    ) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # --- Stage 2: Sentence reranking ---

    def _rerank_candidates(
        self,
        query: str,
        candidates: List[str]
    ) -> List[tuple[int, float]]:
        """Rerank sentence windows using embedding similarity."""
        if not candidates:
            return []

        # Get the query embedding once
        query_emb = self._get_query_embedding(query)

        # Get embeddings for all candidates and compute similarity scores
        scored = []
        for i, cand in enumerate(candidates):
            emb = self._get_query_embedding(cand)
            score = self._cosine_similarity(query_emb, emb)
            scored.append((i, score))

        return sorted(scored, key=lambda x: x[1], reverse=True)

    def search_sentences_in_section(
        self,
        query: str,
        section_id: str,
        top_k: int = 5
    ) -> List[tuple[str, float]]:
        """Find relevant sentence windows within a section."""
        windows = self.vector_store.get_sentence_windows(section_id)
        if not windows:
            return []

        candidates = [w.text for w in windows]
        ranked = self._rerank_candidates(query, candidates)

        results = []
        for idx, score in ranked[:top_k]:
            results.append((windows[idx].id, score))

        return results

    def _create_sentence_windows_on_demand(
        self, 
        section_text: str, 
        section_id: str,
        window_size: int = 3,
        overlap: int = 2
    ) -> List[SentenceWindow]:
        """Create sliding window of overlapping sentences from section text on-demand.
        
        Creates windows where each subsequent window shares (window_size - overlap)
        sentences with the previous one. This is used when sentence windows haven't 
        been pre-embedded.
        
        Args:
            section_text: The text content of the section
            section_id: The ID of the section
            window_size: Number of sentences per window (default 3)
            overlap: Number of overlapping sentences between windows (default 2,
                     meaning windows slide by 1 sentence each time for maximum overlap)
        
        Returns:
            List of SentenceWindow objects with overlapping content.
        """
        import re
        # Split into sentences using common delimiters
        sentence_list = re.split(r"(?<=[.!?])\s+", section_text)
        sentence_list = [s.strip() for s in sentence_list if s.strip()]
        
        windows: List[SentenceWindow] = []
        step = max(1, window_size - overlap)  # How many sentences to advance
        
        # Create sliding overlapping windows of sentences
        for i in range(0, len(sentence_list) - window_size + 1, step):
            window_sentences = sentence_list[i:i + window_size]
            window_text = " ".join(window_sentences)
            
            window_id = f"{section_id}_win_{i}"
            
            windows.append(SentenceWindow(
                id=window_id,
                section_id=section_id,
                window_index=i,
                sentences=window_sentences,
                text=window_text,
                is_embedded=False
            ))
        
        # Handle case where we have some sentences but less than full window_size
        if not windows and sentence_list:
            windows.append(SentenceWindow(
                id=f"{section_id}_win_0",
                section_id=section_id,
                window_index=0,
                sentences=sentence_list,
                text=section_text,
                is_embedded=False
            ))
        
        return windows

    def _embed_and_search_sentences(
        self,
        query: str,
        section_text: str,
        section_id: str,
        document_key: str,
        top_k: int = 5
    ) -> List[tuple[str, float]]:
        """Create sentence windows on-demand, embed them, and search.
        
        This enables two-stage search even when sentences weren't pre-embedded.
        Returns tuples of (window_text, score) instead of just window_id.
        """
        # Create sliding overlapping window of 3 sentences from section text
        windows = self._create_sentence_windows_on_demand(section_text, section_id)
        
        if not windows:
            return []
        
        # Embed all sentence windows
        candidates = [w.text for w in windows]
        candidate_embeddings = self.embed_batch(candidates)
        
        # Get query embedding once
        query_emb = self._get_query_embedding(query)
        
        # Compute similarity scores and sort - store text not id
        scored = []
        for i, (window, emb) in enumerate(zip(windows, candidate_embeddings)):
            score = self._cosine_similarity(query_emb, emb)
            scored.append((windows[i].text, score))
        
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            emb = self._get_query_embedding(text)
            embeddings.append(emb)
        return embeddings

    # --- Full two-stage search ---

    def search(
        self,
        query: str,
        document_key: Optional[str] = None,
        top_sections: int = 5,
        top_sentences_per_section: int = 3
    ) -> List[SearchResult]:
        """
        Two-stage RAG search:
        1. Find relevant sections using embeddings
        2. Within each section, find or create sentence windows and rerank them
        
        If sentences weren't pre-embedded, they are created on-demand from the
        section text and embedded dynamically.
        """
        # Stage 1: Find top sections
        if document_key:
            # Search within specific document
            all_sections = self.vector_store.get_all_sections(document_key)
            query_embedding = self._get_query_embedding(query)

            scored_sections = []
            for section in all_sections:
                emb = self.embed_text_for_search(section.text)
                score = self._cosine_similarity(query_embedding, emb)
                scored_sections.append((section.id, score))

            section_results = sorted(
                scored_sections,
                key=lambda x: x[1],
                reverse=True
            )[:top_sections]
        else:
            # Search across all documents - get sections with their document keys
            query_embedding = self._get_query_embedding(query)
            section_ids, embeddings = self.vector_store.search_sections(
                query_embedding, top_k=top_sections * 2  # Get more to filter by score
            )
            
            if not section_ids:
                return []
                
            # Build results with document keys from metadata
            section_results = []
            for i, section_id in enumerate(section_ids):
                section = self.vector_store.get_section(section_id)
                if section and i < len(embeddings) and embeddings[i]:
                    score = self._cosine_similarity(query_embedding, embeddings[i])
                    # Store tuple of (section_id, document_key, score)
                    section_results.append((section.id, section.document_id, score))

            # Sort by score and take top k
            section_results.sort(key=lambda x: x[2], reverse=True)
            section_results = [(sid, doc_key, score) for sid, doc_key, score in section_results[:top_sections]]

        if not section_results:
            return []

        # Stage 2: Find sentence candidates from each section (using embeddings for filtering)
        # Collect all candidates with their metadata before reranking
        sentence_candidates: List[tuple] = []  # (text, document_key, section_title, relevance_score)
        
        for result_item in section_results:
            if len(result_item) == 3:
                section_id, document_key, section_score = result_item
            else:
                section_id, section_score = result_item
                document_key = None
            
            # Get the full section to access text and metadata
            section = self.vector_store.get_section(section_id)
            if not section:
                continue

            # Get all sentence windows for this section (pre-embedded)
            windows = self.vector_store.get_sentence_windows(section_id)

            if not windows:
                # No pre-embedded sentences - create on-demand and search
                matched_windows = self._embed_and_search_sentences(
                    query=query,
                    section_text=section.text,
                    section_id=section_id,
                    document_key=document_key or "",
                    top_k=top_sentences_per_section
                )
                
                # Add candidates from this section (all of them for reranking)
                for window_text, emb_score in matched_windows:
                    sentence_candidates.append((
                        window_text,
                        document_key or "",
                        section.title,
                        section_score,  # Section-level relevance score
                        emb_score       # Sentence embedding similarity score
                    ))
            else:
                # Use pre-embedded sentence windows - filter with embeddings first
                candidates = [w.text for w in windows]
                ranked = self._rerank_candidates(query, candidates)

                # Add top candidates from this section (for reranking later)
                for idx, emb_score in ranked[:top_sentences_per_section]:
                    sentence_candidates.append((
                        windows[idx].text,
                        document_key or "",
                        f"{section.title}",
                        section_score,  # Section-level relevance score
                        emb_score       # Sentence embedding similarity score
                    ))

        if not sentence_candidates:
            return []

        search_results = []
        for text, doc_key, section_title, section_score, emb_score in sentence_candidates:
            search_results.append(SearchResult(
                text=text,
                document_title="",
                section_title=section_title,
                zotero_key=doc_key,
                relevance_score=emb_score,
                rerank_score=emb_score  # Use same score for both
            ))

        # Sort by embedding score and return top results
        search_results.sort(key=lambda x: x.rerank_score, reverse=True)
        return search_results[:top_sections * top_sentences_per_section]

    def embed_text_for_search(self, text: str) -> List[float]:
        """Generate embedding for search (used internally)."""
        response = ollama.embeddings(
            model=self.config.EMBEDDING_MODEL,
            prompt=text,
            options=self._get_embedding_options()
        )
        return response["embedding"]

    # --- Utility methods ---

    def get_stats(self) -> dict:
        """Get embedding statistics."""
        return {
            "total_sections": self.vector_store.get_section_count(),
            "total_sentence_windows": self.vector_store.get_sentence_count(),
            "embedded_documents": len(self.vector_store.get_embedded_documents())
        }

    def search_best_sentences(
        self,
        query: str,
        document_key: Optional[str] = None,
        top_sections: int = 5,
        top_sentences: int = 10,
        ensure_sentence_embeddings: bool = True,
    ) -> List[SearchResult]:
        """Sentence-focused search.

        Contract:
        - Input: a query sentence/string.
        - Stage 1: embed query and retrieve best-matching sections.
        - Stage 2: for those sections, embed+persist sentence windows (if missing),
          then search sentences via the sentence vector index.
        - Output: top `top_sentences` sentence windows as SearchResult.

        Notes:
        - Sentence windows are stored persistently in Chroma (VectorStore.sentences_collection).
        - If `ensure_sentence_embeddings` is False, we only search what already exists.
        """
        query_embedding = self._get_query_embedding(query)

        # --- Stage 1: pick top sections ---
        # NOTE: In the vector store, section metadata stores "document_key" which is
        # the PDF filename stem (Path.stem) used during embedding. This is NOT the
        # Zotero item key. Zotero item key mapping is handled elsewhere.

        # --- Stage 1: pick top sections ---
        if document_key:
            all_sections = self.vector_store.get_all_sections(document_key)
            if not all_sections:
                return []

            scored_sections: list[tuple[str, str, float]] = []
            for section in all_sections:
                # Sections are already embedded in the store; but we don't have direct access
                # to their embeddings here, so we use an on-the-fly embedding for scoring.
                # (The primary use-case is global search below.)
                emb = self.embed_text_for_search(section.text)
                score = self._cosine_similarity(query_embedding, emb)
                scored_sections.append((section.id, section.document_id, score))

            scored_sections.sort(key=lambda x: x[2], reverse=True)
            section_results = scored_sections[:top_sections]
        else:
            section_ids, embeddings = self.vector_store.search_sections(
                query_embedding, top_k=max(1, top_sections * 2)
            )
            if not section_ids:
                return []

            section_results: list[tuple[str, str, float]] = []
            for i, section_id in enumerate(section_ids):
                section = self.vector_store.get_section(section_id)
                if not section:
                    continue
                if i >= len(embeddings) or not embeddings[i]:
                    continue
                score = self._cosine_similarity(query_embedding, embeddings[i])
                section_results.append((section.id, section.document_id, score))

            section_results.sort(key=lambda x: x[2], reverse=True)
            section_results = section_results[:top_sections]

        if not section_results:
            return []

        # --- Stage 2: ensure sentence embeddings exist for the chosen sections ---
        if ensure_sentence_embeddings:
            windows_to_add: list[SentenceWindow] = []
            for section_id, doc_key, _score in section_results:
                existing = self.vector_store.get_sentence_windows(section_id)
                if existing:
                    continue

                section = self.vector_store.get_section(section_id)
                if not section:
                    continue

                created = self._create_sentence_windows_on_demand(
                    section_text=section.text,
                    section_id=section_id,
                )
                windows_to_add.extend(created)

            if windows_to_add:
                embeddings_to_add = self.embed_batch([w.text for w in windows_to_add])

                for section_id, doc_key, _score in section_results:
                    section_windows: list[SentenceWindow] = [w for w in windows_to_add if w.section_id == section_id]
                    if not section_windows:
                        continue
                    section_embeddings: list[List[float]] = [
                        embeddings_to_add[i]
                        for i, w in enumerate(windows_to_add)
                        if w.section_id == section_id
                    ]

                    # Pull Zotero keys stored for this document (if available)
                    key_meta = self.vector_store.get_document_zotero_keys(doc_key)
                    self.vector_store.add_sentence_windows(
                        section_windows,
                        section_embeddings,
                        document_key=doc_key or "",
                        zotero_key=key_meta.get("zotero_key") or None,
                        parent_item_key=key_meta.get("parent_item_key") or None,
                    )

        # --- Stage 3: search sentence windows within selected documents ---
        # We search per-document to avoid needing a section filter at the vector store layer.
        # IMPORTANT: section.document_id is the Zotero item key (attachment or parent),
        # while the vector store metadata document_key is the PDF filename stem.
        # We therefore filter sentence searches by the Zotero key actually stored
        # in sentence metadata.
        by_doc_keys: list[str] = sorted({doc_key for _sid, doc_key, _s in section_results if doc_key})
        if not by_doc_keys:
            return []

        sentence_hits: list[tuple[str, str, float]] = []  # (window_id, doc_key, score)
        per_doc_k = max(20, top_sentences * 5)

        for doc_key in by_doc_keys:
            ids, embeds = self.vector_store.search_sentences(
                query_embedding=query_embedding,
                document_key=doc_key,
                top_k=per_doc_k,
            )
            for wid, emb in zip(ids, embeds):
                if not emb:
                    continue
                score = self._cosine_similarity(query_embedding, emb)
                sentence_hits.append((wid, doc_key, score))

        if not sentence_hits:
            return []

        sentence_hits.sort(key=lambda x: x[2], reverse=True)
        sentence_hits = sentence_hits[: max(1, top_sentences * 3)]

        # Map window_id -> text and section title
        # We can recover the section_id from the window_id prefix: "{section_id}_win_{i}".
        results: list[SearchResult] = []
        for wid, doc_key, score in sentence_hits:
            section_id = wid.split("_win_")[0] if "_win_" in wid else ""
            section = self.vector_store.get_section(section_id) if section_id else None
            section_title = section.title if section else ""

            # Fetch window text from stored windows list
            text = ""
            for w in self.vector_store.get_sentence_windows(section_id):
                if w.id == wid:
                    text = w.text
                    break

            if not text:
                continue

            results.append(
                SearchResult(
                    text=text,
                    document_title="",
                    section_title=section_title,
                    zotero_key=doc_key,  # doc_key is the Zotero key
                    relevance_score=score,
                    rerank_score=score,
                )
            )

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:top_sentences]
