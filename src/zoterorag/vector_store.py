"""LanceDB-backed vector store with persistence tracking."""

import json
import logging
import os
import threading
import traceback
from pathlib import Path
from typing import Optional, List, Any

import lancedb

from .models import Sentence

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed internal state logs


class VectorStore:
    """LanceDB wrapper for storing and retrieving sentence embeddings."""

    _TABLE_NAME = "sentences"
    _SUPPORTED_INDEX_TYPES = {"IVF_HNSW_SQ", "IVF_RQ", "IVF_PQ", "IVF_FLAT"}

    def __init__(self, persist_directory: str = "./data/vector_store"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._embedded_docs_lock = threading.Lock()
        self._table_lock = threading.Lock()

        # Keep Lance artifacts in a dedicated subdirectory.
        self.db_directory = self.persist_directory / "lancedb"
        self.db_directory.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_directory))

        self.sentences_table = self._open_table_if_exists()
        self._index_ready = False
        self._index_type = self._resolve_index_type()

        self._detected_sentence_dim: int | None = None
        self._detect_dimensions()
        self.index_lock = threading.Lock()


    def _open_table_if_exists(self):
        try:
            return self.db.open_table(self._TABLE_NAME)
        except Exception:
            return None

    @staticmethod
    def _sql_quote(value: str) -> str:
        return value.replace("'", "''")

    @classmethod
    def _where_eq(cls, column: str, value: str) -> str:
        return f"{column} = '{cls._sql_quote(value)}'"

    @classmethod
    def _where_in(cls, column: str, values: List[str]) -> str:
        escaped = ", ".join(f"'{cls._sql_quote(v)}'" for v in values)
        return f"{column} IN ({escaped})"

    def _resolve_index_type(self) -> str:
        configured = os.getenv("LANCEDB_INDEX_TYPE", "IVF_HNSW_SQ")
        index_type = str(configured).strip().upper()
        if index_type not in self._SUPPORTED_INDEX_TYPES:
            logger.warning(
                "Unsupported LANCEDB_INDEX_TYPE=%r. Falling back to IVF_FLAT.",
                configured,
            )
            return "IVF_FLAT"
        logger.info("Using LanceDB index type: %s", index_type)
        return index_type

    def _refresh_table_handle(self) -> None:
        """Reopen table to observe the latest committed version from disk."""
        with self._table_lock:
            self.sentences_table = self._open_table_if_exists()

    def _ensure_cosine_index(self) -> None:
        try:
            self.index_lock.acquire_lock()
            if not self.sentences_table or self._index_ready:
                try:
                    self.sentences_table.optimize()
                except Exception as e:
                    logger.debug("Could not optimize LanceDB table: %s", e)
            try:
                #check if there is already an index on the vector column
                existing_indexes = self.sentences_table.list_indices()
                for idx in existing_indexes:
                    if "vector" in idx.columns and idx.name == "vector_idx":
                        self._index_ready = True
                        return

                self.sentences_table.create_index(
                    metric="cosine",
                    vector_column_name="vector",
                    replace=False,
                    index_type=self._index_type,
                )
                self._index_ready = True
            except Exception as e:
                logger.debug("Could not create LanceDB cosine index yet: %s", e)
                logger.debug("Probably the table is not ready; it will be retried on next add/search.")
        finally:
            self.index_lock.release_lock()


    def _detect_dimensions(self):
        """Detect embedding dimensions from existing table rows."""
        if not self.sentences_table:
            return

        try:
            rows = self.sentences_table.search().select(["vector"]).limit(1).to_list()
            if not rows:
                return

            first = rows[0].get("vector")
            if not first:
                return

            dim = len(first)
            if dim > 0:
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
        batch_size: int = 5000,
    ):
        """Add sentence embeddings to the store.

        Args:
            sentences: List of Sentence objects to add.
            embeddings: Corresponding embedding vectors.
            document_key: The document key these sentences belong to.
            batch_size: Maximum number of items per add operation.
        """
        if not sentences or not embeddings:
            return

        n = min(len(sentences), len(embeddings))
        if n <= 0:
            return
        if len(sentences) != len(embeddings):
            logger.warning(
                "Sentence/embedding length mismatch (%s vs %s); truncating to %s",
                len(sentences),
                len(embeddings),
                n,
            )

        def _clip_list(v: list, limit: int = 50) -> list:
            return list(v[:limit]) if v else []

        for i in range(0, n, batch_size):
            chunk_sentences = sentences[i : i + batch_size]
            chunk_embeddings = embeddings[i : i + batch_size]

            rows: list[dict] = []
            for s, emb in zip(chunk_sentences, chunk_embeddings):
                row = {
                    "id": str(s.id),
                    "vector": [float(x) for x in emb],
                    "document": str(s.text),
                    "document_key": str(document_key),
                    "page": int(s.page),
                    "page_section": int(s.page_section) if s.page_section is not None else None,
                    "sentence_index": int(s.sentence_index),
                }

                citation_numbers = _clip_list([int(num) for num in (s.citation_numbers or [])])
                referenced_texts = _clip_list([str(t) for t in (s.referenced_texts or [])], limit=20)
                referenced_bibtex = _clip_list([str(b) for b in (s.referenced_bibtex or [])], limit=20)

                # Keep schema stable by always writing list fields, even when empty.
                row["citation_numbers"] = citation_numbers
                row["referenced_texts"] = referenced_texts
                row["referenced_bibtex"] = referenced_bibtex

                rows.append(row)

            with self._table_lock:
                if self.sentences_table is None:
                    self.sentences_table = self.db.create_table(
                        self._TABLE_NAME,
                        data=rows,
                        mode="overwrite",
                    )
                    self._detected_sentence_dim = len(rows[0].get("vector") or []) if rows else None
                else:
                    ids = [r["id"] for r in rows]
                    try:
                        self.sentences_table.delete(self._where_in("id", ids))
                    except Exception:
                        # If delete fails, add still proceeds; consumers should dedupe by id if needed.
                        logger.debug("Could not delete existing ids before add")
                    self.sentences_table.add(rows)

        self._ensure_cosine_index()
        self._refresh_table_handle()

    def get_sentences(self, document_key: str) -> List[Sentence]:
        if not self.sentences_table:
            return []

        try:
            rows = (
                self._safe_select(
                    self.sentences_table.search().where(self._where_eq("document_key", document_key)),
                    [
                        "id",
                        "document",
                        "document_key",
                        "page",
                        "page_section",
                        "sentence_index",
                        "citation_numbers",
                        "referenced_texts",
                        "referenced_bibtex",
                    ],
                )
                .to_list()
            )
        except Exception:
            return []

        rows.sort(key=lambda r: (int(r.get("page", 1) or 1), int(r.get("sentence_index", 0) or 0)))

        out: list[Sentence] = []
        for row in rows:
            out.append(
                Sentence(
                    id=str(row.get("id", "")),
                    document_id=str(row.get("document_key", document_key)),
                    page=int(row.get("page", 1) or 1),
                    page_section=(
                        int(row["page_section"]) if row.get("page_section") is not None else None
                    ),
                    sentence_index=int(row.get("sentence_index", 0) or 0),
                    text=str(row.get("document", "")),
                    is_embedded=True,
                    citation_numbers=list(row.get("citation_numbers") or []),
                    referenced_texts=list(row.get("referenced_texts") or []),
                    referenced_bibtex=list(row.get("referenced_bibtex") or []),
                )
            )
        return out

    def get_sentence_metadatas_by_ids(self, ids: List[str]) -> dict[str, dict]:
        """Fetch sentence metadatas for a small set of ids."""
        if not ids or not self.sentences_table:
            return {}

        out: dict[str, dict] = {}
        for sid in [str(i) for i in ids]:
            try:
                row_list = (
                    self._safe_select(
                        self.sentences_table.search().where(self._where_eq("id", sid)).limit(1),
                        [
                            "id",
                            "document_key",
                            "page",
                            "page_section",
                            "sentence_index",
                            "citation_numbers",
                            "referenced_texts",
                            "referenced_bibtex",
                        ],
                    )
                    .to_list()
                )
            except Exception:
                continue
            if not row_list:
                continue
            row = row_list[0]
            out[sid] = {
                "document_key": row.get("document_key"),
                "page": row.get("page"),
                "page_section": row.get("page_section"),
                "sentence_index": row.get("sentence_index"),
                "citation_numbers": list(row.get("citation_numbers") or []),
                "referenced_texts": list(row.get("referenced_texts") or []),
                "referenced_bibtex": list(row.get("referenced_bibtex") or []),
            }
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
        ids, _scores, _metadatas = self.search_sentence_ids(
            query_embedding=query_embedding,
            document_key=document_key,
            top_k=top_k,
        )
        return ids, []

    def search_sentence_ids(
        self,
        query_embedding: List[float],
        document_key: Optional[str] = None,
        top_k: int = 10,
        include_documents: bool = False,
    ) -> tuple[List[str], List[float], List[dict]]:
        """Search sentence vectors efficiently.

        Returns:
            (ids, relevance_scores, metadatas)
        """
        if not self.sentences_table:
            return [], [], []

        #self._ensure_cosine_index()

        def _execute_query() -> list[dict]:
            builder: Any = self.sentences_table.search([float(x) for x in query_embedding])
            builder = builder.metric("cosine")
            if document_key:
                builder = builder.where(self._where_eq("document_key", document_key))

            columns = [
                "id",
                "document_key",
                "page",
                "page_section",
                "sentence_index",
                "citation_numbers",
                "referenced_texts",
                "referenced_bibtex",
            ]
            if include_documents:
                columns.append("document")

            return self._safe_select(
                builder.limit(max(1, int(top_k))),
                columns,
            ).to_list()

        rows: list[dict]
        try:
            rows = _execute_query()
        except Exception:
            # A long-lived table handle can lag behind latest writes; refresh and retry once.
            self._refresh_table_handle()
            if not self.sentences_table:
                return [], [], []
            try:
                rows = _execute_query()
            except Exception:
                return [], [], []

        ids: list[str] = []
        scores: list[float] = []
        metadatas: list[dict] = []

        for row in rows:
            sid = str(row.get("id", ""))
            if not sid:
                continue

            distance = float(row.get("_distance", 1.0))
            score = 1.0 - (distance/2)  # Convert cosine distance to similarity score

            meta = {
                "document_key": row.get("document_key"),
                "page": row.get("page"),
                "page_section": row.get("page_section"),
                "sentence_index": row.get("sentence_index"),
                "citation_numbers": list(row.get("citation_numbers") or []),
                "referenced_texts": list(row.get("referenced_texts") or []),
                "referenced_bibtex": list(row.get("referenced_bibtex") or []),
            }
            if include_documents:
                meta["document"] = str(row.get("document", ""))

            ids.append(sid)
            scores.append(score)
            metadatas.append(meta)

        return ids, scores, metadatas

    def get_sentence_texts_by_ids(self, ids: List[str]) -> dict[str, str]:
        """Fetch sentence texts for a small set of ids."""
        if not ids or not self.sentences_table:
            return {}

        out: dict[str, str] = {}
        for sid in [str(i) for i in ids]:
            try:
                row_list = (
                    self._safe_select(
                        self.sentences_table.search().where(self._where_eq("id", sid)).limit(1),
                        ["id", "document"],
                    )
                    .to_list()
                )
            except Exception:
                continue
            if not row_list:
                continue
            row = row_list[0]
            doc = row.get("document")
            if doc is not None:
                out[sid] = str(doc)
        return out

    def is_document_embedded(self, document_key: str) -> bool:
        """Return True if at least one sentence vector exists for this document."""
        if not document_key or not self.sentences_table:
            return False

        try:
            rows = (
                self._safe_select(
                    self.sentences_table.search().where(self._where_eq("document_key", document_key)).limit(1),
                    ["id"],
                )
                .to_list()
            )
            return bool(rows)
        except Exception:
            return False

    # --- Cleanup ---

    def delete_document(self, document_key: str):
        if not self.sentences_table:
            return
        try:
            self.sentences_table.delete(self._where_eq("document_key", document_key))
            self._refresh_table_handle()
        except Exception:
            logger.debug("Failed to delete document %s from LanceDB", document_key)

    def clear_all(self):
        with self._table_lock:
            try:
                self.db.drop_table(self._TABLE_NAME)
            except Exception:
                pass
            self.sentences_table = None
            self._detected_sentence_dim = None
            self._index_ready = False

    def get_sentence_count(self) -> int:
        if not self.sentences_table:
            return 0
        try:
            return int(self.sentences_table.count_rows())
        except Exception:
            return 0

    def get_document_title(self, document_key: str) -> Optional[str]:
        # Titles are no longer stored in the vector DB; keep API for MCP compatibility.
        return None

    def _available_columns(self) -> set[str]:
        if not self.sentences_table:
            return set()
        try:
            return set(self.sentences_table.schema.names)
        except Exception:
            return set()

    def _existing_columns(self, requested: List[str]) -> List[str]:
        available = self._available_columns()
        if not available:
            return requested
        return [c for c in requested if c in available]

    def _safe_select(self, builder: Any, requested_columns: List[str]) -> Any:
        columns = self._existing_columns(requested_columns)
        if columns:
            return builder.select(columns)
        return builder
