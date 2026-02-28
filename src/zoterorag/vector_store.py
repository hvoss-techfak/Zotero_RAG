"""ChromaDB-based vector store with persistence tracking."""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional, List

import chromadb
from chromadb.config import Settings

from .models import Sentence

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB wrapper for storing and retrieving sentence embeddings."""

    def __init__(self, persist_directory: str = "./data/vector_store"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._embedded_docs_lock = threading.Lock()

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # Single collection: sentences
        self.sentences_collection = self._get_or_create_collection("sentences")

        self._detected_sentence_dim: int | None = None
        self._detect_dimensions()

    def _detect_dimensions(self):
        """Detect embedding dimensions from existing collections."""
        try:
            result = self.sentences_collection.get(limit=1, include=["embeddings"])
            if result and result.get("embeddings") and result.get("ids"):
                emb_array = result["embeddings"]
                if not emb_array or emb_array[0] is None:
                    return
                first = emb_array[0]

                # Ignore mocks / placeholder objects that don't represent real embeddings.
                if type(first).__module__.startswith("unittest."):
                    return

                dim: int | None = None
                if hasattr(first, "shape") and getattr(first, "shape", None):
                    try:
                        dim = int(first.shape[0])
                    except Exception:
                        dim = None
                elif isinstance(first, list):
                    dim = len(first)
                else:
                    try:
                        dim = len(first)
                    except TypeError:
                        dim = None

                if dim and dim > 0:
                    self._detected_sentence_dim = dim
                    logger.info(
                        "Detected existing sentence embedding dimension: %s",
                        self._detected_sentence_dim,
                    )

        except Exception as e:
            logger.debug("Could not detect sentence dimensions: %s", e)

    def get_detected_dimension(self) -> int | None:
        """Return detected sentence embedding dimension, if any."""
        return self._detected_sentence_dim

    def has_dimension_mismatch(self, expected_dim: int) -> bool:
        if self._detected_sentence_dim is None:
            return False
        return self._detected_sentence_dim != expected_dim

    def _get_or_create_collection(self, name: str):
        """Get or create a collection (compatible with older tests/client mocks)."""
        # Some tests mock `get_or_create_collection`, others mock `get_collection/create_collection`.
        if hasattr(self.client, "get_or_create_collection"):
            return self.client.get_or_create_collection(name)

        try:
            return self.client.get_collection(name)
        except Exception:
            return self.client.create_collection(name)

    # --- Document metadata tracking ---

    def get_embedded_documents(self) -> dict[str, int]:
        """Return dict mapping document keys to their sentence count."""
        meta_path = self.persist_directory / "embedded_docs.json"
        with self._embedded_docs_lock:
            if meta_path.exists():
                try:
                    with open(meta_path, "r") as f:
                        content = f.read().strip()
                        if content:
                            return json.loads(content)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(
                        "Failed to load embedded_docs.json: %s. Starting fresh.",
                        e,
                    )
                    backup_path = meta_path.with_suffix(".json.bak")
                    try:
                        meta_path.rename(backup_path)
                    except OSError:
                        pass
            return {}

    def save_embedded_documents(self, docs: dict[str, int], allow_empty: bool = True):
        if not docs and not allow_empty:
            return

        meta_path = self.persist_directory / "embedded_docs.json"

        with self._embedded_docs_lock:
            tmp_path = meta_path.with_suffix(".json.tmp")
            try:
                with open(tmp_path, "w") as f:
                    json.dump(docs, f, sort_keys=True)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, meta_path)
            except IOError as e:
                logger.error("Failed to save embedded_docs.json: %s", e)
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass

    def update_embedded_document(self, document_key: str, sentence_count: int) -> None:
        meta_path = self.persist_directory / "embedded_docs.json"

        with self._embedded_docs_lock:
            docs: dict[str, int] = {}
            if meta_path.exists():
                try:
                    content = meta_path.read_text().strip()
                    if content:
                        docs = json.loads(content)
                except (json.JSONDecodeError, OSError):
                    docs = {}

            docs[document_key] = sentence_count

            tmp_path = meta_path.with_suffix(".json.tmp")
            try:
                with open(tmp_path, "w") as f:
                    json.dump(docs, f, sort_keys=True)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, meta_path)
            except OSError as e:
                logger.error("Failed to update embedded_docs.json: %s", e)
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass

    # --- Sentence operations ---

    def add_sentences(
        self,
        sentences: List[Sentence],
        embeddings: List[List[float]],
        document_key: str,
    ):
        """Add sentence embeddings to the store."""
        if not sentences or not embeddings:
            return

        ids = [s.id for s in sentences]
        documents = [s.text for s in sentences]
        metadatas = [
            {
                "document_key": document_key,
                "page": int(s.page),
                "page_section": int(s.page_section) if s.page_section is not None else None,
                "sentence_index": int(s.sentence_index),
            }
            for s in sentences
        ]

        self.sentences_collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def get_sentences(self, document_key: str) -> List[Sentence]:
        result = self.sentences_collection.get(where={"document_key": document_key})
        if not result.get("ids"):
            return []

        out: list[Sentence] = []
        for i, sid in enumerate(result["ids"]):
            meta = result["metadatas"][i] or {}
            out.append(
                Sentence(
                    id=sid,
                    document_id=document_key,
                    page=int(meta.get("page", 1)),
                    page_section=(
                        int(meta["page_section"]) if meta.get("page_section") is not None else None
                    ),
                    sentence_index=int(meta.get("sentence_index", 0)),
                    text=result["documents"][i],
                    is_embedded=True,
                )
            )
        return out

    def search_sentences(
        self,
        query_embedding: List[float],
        document_key: str,
        top_k: int = 10,
    ) -> tuple[List[str], List[List[float]]]:
        """Backward-compatible search API.

        NOTE: For large-scale search you should prefer :meth:`search_sentence_ids` which
        avoids returning embeddings to Python.
        """
        ids, _distances, _metadatas = self.search_sentence_ids(
            query_embedding=query_embedding,
            document_key=document_key,
            top_k=top_k,
        )
        # Older callers expected embeddings; returning empty keeps behavior safe and
        # avoids loading vector payloads into memory.
        return ids, []

    def search_sentence_ids(
        self,
        query_embedding: List[float],
        document_key: Optional[str] = None,
        top_k: int = 10,
        include_documents: bool = False,
    ) -> tuple[List[str], List[float], List[dict]]:
        """Search sentence vectors efficiently.

        This method uses Chroma's persistent ANN index and returns only ids + distances
        (and metadatas). It does *not* include embeddings, so it remains fast and
        memory-efficient for millions of rows.

        Returns:
            (ids, distances, metadatas)
        """
        where = {"document_key": document_key} if document_key else None
        include: list[str] = ["distances", "metadatas"]
        if include_documents:
            include.append("documents")

        results = self.sentences_collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, int(top_k)),
            where=where,
            include=include,
        )

        if not results or not results.get("ids") or not results["ids"][0]:
            return [], [], []

        ids: list[str] = list(results["ids"][0])
        distances_raw = results.get("distances")
        distances: list[float]
        if distances_raw and distances_raw[0] is not None:
            distances = [float(d) for d in distances_raw[0]]
        else:
            distances = []

        metadatas_raw = results.get("metadatas")
        metadatas: list[dict]
        if metadatas_raw and metadatas_raw[0] is not None:
            metadatas = [m or {} for m in metadatas_raw[0]]
        else:
            metadatas = [{} for _ in ids]

        return ids, distances, metadatas

    def get_sentence_texts_by_ids(self, ids: List[str]) -> dict[str, str]:
        """Fetch sentence texts for a small set of ids."""
        if not ids:
            return {}

        # Chroma returns lists aligned with the input ids.
        res = self.sentences_collection.get(ids=ids, include=["documents"])
        out: dict[str, str] = {}
        res_ids = res.get("ids") or []
        docs = res.get("documents") or []
        for sid, doc in zip(res_ids, docs):
            if sid and doc is not None:
                out[str(sid)] = str(doc)
        return out

    # --- Cleanup ---

    def delete_document(self, document_key: str):
        self.sentences_collection.delete(where={"document_key": document_key})

    def clear_all(self):
        self.client.delete_collection("sentences")
        self.sentences_collection = self._get_or_create_collection("sentences")

    def get_sentence_count(self) -> int:
        cnt = getattr(self.sentences_collection, "count", 0)
        return cnt() if callable(cnt) else int(cnt)

    def get_document_title(self, document_key: str) -> Optional[str]:
        # Titles are no longer stored in the vector DB; keep API for MCP compatibility.
        return None

