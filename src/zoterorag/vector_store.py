"""ChromaDB-based vector store with persistence tracking."""

import logging
from pathlib import Path
from typing import Optional, List
import json

import chromadb
from chromadb.config import Settings

from .models import Section, SentenceWindow


logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB wrapper for storing and retrieving embeddings."""

    def __init__(self, persist_directory: str = "./data/vector_store"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )

        # Collections
        self.sections_collection = self._get_or_create_collection("sections")
        self.sentences_collection = self._get_or_create_collection("sentences")
        
        # Detect existing embedding dimensions from collections
        self._detected_section_dim: int | None = None
        self._detected_sentence_dim: int | None = None
        self._detect_dimensions()

    def _detect_dimensions(self):
        """Detect embedding dimensions from existing collections."""
        import numpy as np
        
        try:
            # Try to get a sample embedding to detect dimension
            result = self.sections_collection.get(limit=1, include=["embeddings"])
            if result and result.get("embeddings"):
                emb_array = result["embeddings"]
                # Handle both list and numpy array, check for non-empty
                if hasattr(emb_array, 'any') and emb_array.any():
                    self._detected_section_dim = int(emb_array[0].shape[0]) if hasattr(emb_array[0], 'shape') else len(emb_array[0])
                    logger.info(f"Detected existing section embedding dimension: {self._detected_section_dim}")
                elif isinstance(emb_array, list) and emb_array and emb_array[0] is not None:
                    self._detected_section_dim = len(emb_array[0])
                    logger.info(f"Detected existing section embedding dimension: {self._detected_section_dim}")
        except Exception as e:
            logger.debug(f"Could not detect section dimensions: {e}")
        
        try:
            result = self.sentences_collection.get(limit=1, include=["embeddings"])
            if result and result.get("embeddings"):
                emb_array = result["embeddings"]
                # Handle both list and numpy array
                if hasattr(emb_array, 'any') and emb_array.any():
                    self._detected_sentence_dim = int(emb_array[0].shape[0]) if hasattr(emb_array[0], 'shape') else len(emb_array[0])
                    logger.info(f"Detected existing sentence embedding dimension: {self._detected_sentence_dim}")
                elif isinstance(emb_array, list) and emb_array and emb_array[0] is not None:
                    self._detected_sentence_dim = len(emb_array[0])
                    logger.info(f"Detected existing sentence embedding dimension: {self._detected_sentence_dim}")
        except Exception as e:
            logger.debug(f"Could not detect sentence dimensions: {e}")

    def get_detected_dimension(self) -> int | None:
        """Return detected section embedding dimension, if any."""
        return self._detected_section_dim

    def has_dimension_mismatch(self, expected_dim: int) -> bool:
        """Check if there's a dimension mismatch with existing embeddings.
        
        Returns True if existing embeddings have different dimension than expected.
        """
        if self._detected_section_dim is None:
            return False  # No existing data to compare
        return self._detected_section_dim != expected_dim

    def _get_or_create_collection(self, name: str):
        """Get or create a collection."""
        try:
            return self.client.get_collection(name)
        except Exception:
            return self.client.create_collection(name)

    # --- Document metadata tracking ---

    def get_embedded_documents(self) -> dict[str, int]:
        """Return dict mapping document keys to their section count."""
        meta_path = self.persist_directory / "embedded_docs.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load embedded_docs.json: {e}. Starting fresh.")
                # Backup corrupted file
                backup_path = meta_path.with_suffix(".json.bak")
                try:
                    meta_path.rename(backup_path)
                    logger.info(f"Backed up corrupted file to {backup_path}")
                except OSError:
                    pass
        return {}

    def save_embedded_documents(self, docs: dict[str, int], allow_empty: bool = True):
        """Save document embedding metadata.
        
        Args:
            docs: Dictionary of document keys to section counts
            allow_empty: If False, don't save empty dictionaries
        """
        if not docs and not allow_empty:
            return
            
        meta_path = self.persist_directory / "embedded_docs.json"
        try:
            with open(meta_path, "w") as f:
                json.dump(docs, f, sort_keys=True)
        except IOError as e:
            logger.error(f"Failed to save embedded_docs.json: {e}")

    # --- Section operations ---

    def add_sections(
        self,
        sections: List[Section],
        embeddings: List[List[float]],
        document_key: str
    ):
        """Add section embeddings to the store."""
        if not sections or not embeddings:
            return

        ids = [s.id for s in sections]
        documents = [s.text for s in sections]
        metadatas = [
            {
                "document_key": document_key,
                "title": s.title,
                "level": str(s.level),
                "start_page": s.start_page,
                "end_page": s.end_page,
            }
            for s in sections
        ]

        self.sections_collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def get_section(self, section_id: str) -> Optional[Section]:
        """Retrieve a section by ID."""
        result = self.sections_collection.get(ids=[section_id])
        if not result["ids"]:
            return None

        idx = 0
        meta = result["metadatas"][idx]
        document_key = meta.get("document_key", "")
        return Section(
            id=section_id,
            text=result["documents"][idx],
            title=meta.get("title", ""),
            level=int(meta.get("level", 1)),
            start_page=int(meta.get("start_page", 0)),
            end_page=int(meta.get("end_page", 0)),
            document_id=document_key
        )

    def get_all_sections(self, document_key: str) -> List[Section]:
        """Get all sections for a document."""
        result = self.sections_collection.get(
            where={"document_key": document_key}
        )
        if not result["ids"]:
            return []

        sections = []
        for i, sid in enumerate(result["ids"]):
            meta = result["metadatas"][i]
            sections.append(Section(
                id=sid,
                text=result["documents"][i],
                title=meta.get("title", ""),
                level=int(meta.get("level", 1)),
                start_page=int(meta.get("start_page", 0)),
                end_page=int(meta.get("end_page", 0)),
                document_id=document_key
            ))
        return sections

    def search_sections(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> tuple[List[str], List[List[float]]]:
        """Search sections by embedding similarity."""
        results = self.sections_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["embeddings", "metadatas", "documents"]
        )

        if not results or not results.get("ids") or not results["ids"][0]:
            return [], []

        ids = results["ids"][0]
        # Handle None embeddings (can happen with dimension mismatches)
        raw_embeddings = results.get("embeddings")
        if raw_embeddings is None:
            embeddings = []
        elif isinstance(raw_embeddings, list) and len(raw_embeddings) > 0:
            first_emb = raw_embeddings[0]
            if first_emb is not None:
                # Convert numpy arrays to lists for compatibility
                import numpy as np
                if hasattr(first_emb, 'tolist'):
                    embeddings = [e.tolist() if hasattr(e, 'tolist') else e for e in first_emb]
                elif isinstance(first_emb, list):
                    embeddings = first_emb
                else:
                    embeddings = []
            else:
                embeddings = []
        else:
            embeddings = []
        return ids, embeddings

    # --- Sentence window operations ---

    def add_sentence_windows(
        self,
        windows: List[SentenceWindow],
        embeddings: List[List[float]],
        document_key: str
    ):
        """Add sentence window embeddings to the store."""
        if not windows or not embeddings:
            return

        ids = [w.id for w in windows]
        documents = [w.text for w in windows]
        metadatas = [
            {
                "document_key": document_key,
                "section_id": w.section_id,
                "window_index": str(w.window_index),
            }
            for w in windows
        ]

        self.sentences_collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def get_sentence_windows(self, section_id: str) -> List[SentenceWindow]:
        """Get all sentence windows for a section."""
        result = self.sentences_collection.get(
            where={"section_id": section_id}
        )
        if not result["ids"]:
            return []

        windows = []
        for i, wid in enumerate(result["ids"]):
            meta = result["metadatas"][i]
            windows.append(SentenceWindow(
                id=wid,
                text=result["documents"][i],
                section_id=section_id,
                window_index=int(meta.get("window_index", 0))
            ))
        return windows

    def search_sentences(
        self,
        query_embedding: List[float],
        document_key: str,
        top_k: int = 10
    ) -> tuple[List[str], List[List[float]]]:
        """Search sentence windows within a document."""
        results = self.sentences_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"document_key": document_key},
            include=["embeddings", "metadatas", "documents"]
        )

        if not results or not results.get("ids") or not results["ids"][0]:
            return [], []

        ids = results["ids"][0]
        # Handle None embeddings (can happen with dimension mismatches)
        raw_embeddings = results.get("embeddings")
        if raw_embeddings is None:
            embeddings = []
        elif isinstance(raw_embeddings, list) and len(raw_embeddings) > 0:
            first_emb = raw_embeddings[0]
            if first_emb is not None:
                # Convert numpy arrays to lists for compatibility
                import numpy as np
                if hasattr(first_emb, 'tolist'):
                    embeddings = [e.tolist() if hasattr(e, 'tolist') else e for e in first_emb]
                elif isinstance(first_emb, list):
                    embeddings = first_emb
                else:
                    embeddings = []
            else:
                embeddings = []
        else:
            embeddings = []
        return ids, embeddings

    # --- Cleanup ---

    def delete_document(self, document_key: str):
        """Delete all vectors for a document."""
        self.sections_collection.delete(where={"document_key": document_key})
        self.sentences_collection.delete(where={"document_key": document_key})

    def clear_all(self):
        """Clear all collections (for testing)."""
        self.client.delete_collection("sections")
        self.client.delete_collection("sentences")
        self.sections_collection = self._get_or_create_collection("sections")
        self.sentences_collection = self._get_or_create_collection("sentences")

    def get_document_title(self, document_key: str) -> Optional[str]:
        """Look up the title for a document from section metadata.
        
        Returns the title of the first section found for this document,
        or None if not found.
        """
        result = self.sections_collection.get(
            where={"document_key": document_key},
            limit=1,
            include=["metadatas"]
        )
        if result and result.get("metadatas"):
            return result["metadatas"][0].get("title", "")
        return None

    def get_section_count(self) -> int:
        """Return total number of embedded sections."""
        return self.sections_collection.count()

    def get_sentence_count(self) -> int:
        """Return total number of embedded sentence windows."""
        return self.sentences_collection.count()
