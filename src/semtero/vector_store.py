"""LanceDB-backed vector store with persistence tracking."""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, List, Literal, Optional

import lancedb
import pyarrow as pa

from .models import Sentence

logger = logging.getLogger(__name__)

IndexType = Literal["IVF_HNSW_SQ", "IVF_RQ", "IVF_PQ", "IVF_FLAT"]


class VectorStore:
    """LanceDB wrapper for storing and retrieving sentence embeddings."""

    _TABLE_NAME = "sentences"
    _SUPPORTED_INDEX_TYPES: set[IndexType] = {
        "IVF_HNSW_SQ",
        "IVF_RQ",
        "IVF_PQ",
        "IVF_FLAT",
    }
    _SENTENCE_COLUMNS = (
        "id",
        "vector",
        "document",
        "document_key",
        "page",
        "page_section",
        "sentence_index",
        "citation_numbers",
        "referenced_texts",
        "referenced_bibtex",
    )

    def __init__(self, persist_directory: str = "./data/vector_store"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._embedded_docs_lock = threading.Lock()
        self._embedded_docs_cache: dict[str, int] | None = None
        self._table_lock = threading.Lock()

        self.db_directory = self.persist_directory / "lancedb"
        self.db_directory.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_directory))

        self.sentences_table = self._open_table_if_exists()
        self._index_ready = False
        self._index_type = self._resolve_index_type()
        self._detected_sentence_dim: int | None = None
        self.index_lock = threading.Lock()

        self._repair_sentence_table_if_needed()
        self._detect_dimensions()

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

    def _resolve_index_type(self) -> IndexType:
        configured = os.getenv("LANCEDB_INDEX_TYPE", "IVF_HNSW_SQ")
        index_type = str(configured).strip().upper()
        if index_type not in self._SUPPORTED_INDEX_TYPES:
            logger.warning(
                "Unsupported LANCEDB_INDEX_TYPE=%r. Falling back to IVF_FLAT.",
                configured,
            )
            return "IVF_FLAT"
        logger.info("Using LanceDB index type: %s", index_type)
        return index_type  # type: ignore[return-value]

    def _refresh_table_handle(self) -> None:
        """Reopen table to observe the latest committed version from disk."""
        with self._table_lock:
            previous_table = self.sentences_table
            refreshed_table = self._open_table_if_exists()
            self.sentences_table = refreshed_table
            if refreshed_table is None:
                self._detected_sentence_dim = None
                self._index_ready = False
                return

            if previous_table is not refreshed_table:
                self._index_ready = False

            schema_dim = self._vector_dimension_from_schema(refreshed_table.schema)
            self._detected_sentence_dim = schema_dim
            if self._detected_sentence_dim is None:
                self._detect_dimensions()

    def _prepare_for_read(self) -> bool:
        """Refresh the table handle before read operations so background writes are visible."""
        self._refresh_table_handle()
        return self.sentences_table is not None

    def _ensure_cosine_index(self) -> None:
        if not self.sentences_table:
            return
        try:
            self.index_lock.acquire_lock()
            if self._index_ready:
                return
            try:
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
                logger.debug(
                    "Probably the table is not ready; it will be retried on next add/search."
                )
        finally:
            self.index_lock.release_lock()

    @classmethod
    def _sentence_schema(cls, dim: int) -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.string(), nullable=False),
                pa.field("vector", pa.list_(pa.float32(), dim), nullable=False),
                pa.field("document", pa.string(), nullable=False),
                pa.field("document_key", pa.string(), nullable=False),
                pa.field("page", pa.int64(), nullable=False),
                pa.field("page_section", pa.int64(), nullable=True),
                pa.field("sentence_index", pa.int64(), nullable=False),
                pa.field("citation_numbers", pa.list_(pa.int64()), nullable=False),
                pa.field("referenced_texts", pa.list_(pa.string()), nullable=False),
                pa.field("referenced_bibtex", pa.list_(pa.string()), nullable=False),
            ]
        )

    @staticmethod
    def _vector_dimension_from_schema(schema: pa.Schema | None) -> int | None:
        if schema is None or "vector" not in schema.names:
            return None
        try:
            vector_type = schema.field("vector").type
        except (KeyError, IndexError):
            return None
        if pa.types.is_fixed_size_list(vector_type):
            return int(vector_type.list_size)
        return None

    @staticmethod
    def _field_is_nullish(field: pa.Field) -> bool:
        field_type = field.type
        if pa.types.is_null(field_type):
            return True
        if (
            pa.types.is_list(field_type)
            or pa.types.is_large_list(field_type)
            or pa.types.is_fixed_size_list(field_type)
        ) and pa.types.is_null(field_type.value_type):
            return True
        return False

    def _sentence_table_repair_reasons(self, schema: pa.Schema | None) -> list[str]:
        if schema is None:
            return []

        reasons: list[str] = []
        missing = [name for name in self._SENTENCE_COLUMNS if name not in schema.names]
        if missing:
            reasons.append(f"missing columns: {', '.join(missing)}")

        nullish = [
            name
            for name in self._SENTENCE_COLUMNS
            if name in schema.names and self._field_is_nullish(schema.field(name))
        ]
        if nullish:
            reasons.append(f"null-typed columns: {', '.join(nullish)}")

        return reasons

    def _normalize_sentence_row(
        self, row: dict, document_key: str | None = None
    ) -> dict:
        page_section = row.get("page_section")
        return {
            "id": str(row.get("id", "")),
            "vector": [float(x) for x in (row.get("vector") or [])],
            "document": str(row.get("document") or ""),
            "document_key": str(document_key or row.get("document_key") or ""),
            "page": int(row.get("page", 1) or 1),
            "page_section": int(page_section) if page_section is not None else None,
            "sentence_index": int(row.get("sentence_index", 0) or 0),
            "citation_numbers": [int(v) for v in (row.get("citation_numbers") or [])],
            "referenced_texts": [str(v) for v in (row.get("referenced_texts") or [])],
            "referenced_bibtex": [str(v) for v in (row.get("referenced_bibtex") or [])],
        }

    @classmethod
    def _rows_to_arrow_table(cls, rows: list[dict], dim: int) -> pa.Table:
        batch = pa.RecordBatch.from_pylist(rows, schema=cls._sentence_schema(dim))
        return pa.Table.from_batches([batch], schema=cls._sentence_schema(dim))

    def _detect_dimensions(self) -> None:
        if not self.sentences_table:
            return

        schema_dim = self._vector_dimension_from_schema(self.sentences_table.schema)
        if schema_dim and schema_dim > 0:
            self._detected_sentence_dim = schema_dim
            logger.info(
                "Detected existing sentence embedding dimension: %s",
                self._detected_sentence_dim,
            )
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
        if self._detected_sentence_dim is None:
            self._refresh_table_handle()
        return self._detected_sentence_dim

    def has_dimension_mismatch(self, expected_dim: int) -> bool:
        if self._detected_sentence_dim is None:
            return False
        return self._detected_sentence_dim != expected_dim

    def _validate_batch_dimensions(self, embeddings: List[List[float]]) -> int:
        dims = sorted({len(emb or []) for emb in embeddings})
        if not dims or dims == [0]:
            raise ValueError("Embedding batch is empty or contains only empty vectors.")
        if len(dims) != 1:
            raise ValueError(
                f"Embedding batch contains inconsistent dimensions: {dims}."
            )

        dim = dims[0]
        existing_dim = self.get_detected_dimension()
        if existing_dim and existing_dim != dim:
            raise ValueError(
                "Embedding dimension mismatch: existing store uses "
                f"{existing_dim} dimensions but this batch has {dim}. "
                "Clear the vector store or re-embed with a consistent model/dimension setting."
            )
        return dim

    def _repair_sentence_table_if_needed(self) -> None:
        if not self.sentences_table:
            return

        schema = self.sentences_table.schema
        reasons = self._sentence_table_repair_reasons(schema)
        if not reasons:
            return

        dim = self._vector_dimension_from_schema(schema)
        if dim is None:
            try:
                rows = (
                    self.sentences_table.search().select(["vector"]).limit(1).to_list()
                )
                first = rows[0].get("vector") if rows else None
                dim = len(first) if first else None
            except Exception:
                dim = None

        if dim is None or dim <= 0:
            logger.warning(
                "Existing LanceDB schema is incompatible (%s) and could not be repaired automatically.",
                "; ".join(reasons),
            )
            return

        try:
            count = int(self.sentences_table.count_rows())
        except Exception:
            count = 0

        if count <= 0:
            logger.warning(
                "Dropping empty LanceDB sentence table with incompatible schema (%s).",
                "; ".join(reasons),
            )
            try:
                self.db.drop_table(self._TABLE_NAME)
            except Exception:
                pass
            self.sentences_table = None
            self._detected_sentence_dim = None
            return

        read_columns = self._existing_columns(list(self._SENTENCE_COLUMNS))
        try:
            existing_rows = (
                self.sentences_table.search()
                .select(read_columns)
                .limit(count)
                .to_list()
            )
            repaired_rows: list[dict] = []
            dropped_rows = 0
            for row in existing_rows:
                normalized = self._normalize_sentence_row(row)
                if not normalized["id"] or len(normalized["vector"]) != dim:
                    dropped_rows += 1
                    continue
                repaired_rows.append(normalized)

            self.db.drop_table(self._TABLE_NAME)
            if repaired_rows:
                self.sentences_table = self.db.create_table(
                    self._TABLE_NAME,
                    data=self._rows_to_arrow_table(repaired_rows, dim),
                    mode="overwrite",
                )
            else:
                self.sentences_table = None
            self._detected_sentence_dim = dim
            self._index_ready = False
            logger.warning(
                "Repaired LanceDB sentence table schema (%s)%s.",
                "; ".join(reasons),
                f" Dropped {dropped_rows} malformed rows" if dropped_rows else "",
            )
        except Exception as e:
            logger.warning(
                "Failed to repair LanceDB sentence table schema (%s): %s",
                "; ".join(reasons),
                e,
            )
            self.sentences_table = self._open_table_if_exists()

    def _ensure_sentence_table_compatible(self, embedding_dim: int) -> None:
        if not self.sentences_table:
            return

        self._repair_sentence_table_if_needed()
        if not self.sentences_table:
            return

        reasons = self._sentence_table_repair_reasons(self.sentences_table.schema)
        if reasons:
            raise ValueError(
                "Existing vector store schema is incompatible ("
                + "; ".join(reasons)
                + "). Clear the vector store directory and re-embed."
            )

        existing_dim = (
            self.get_detected_dimension()
            or self._vector_dimension_from_schema(self.sentences_table.schema)
        )
        if existing_dim and existing_dim != embedding_dim:
            raise ValueError(
                "Embedding dimension mismatch: existing store uses "
                f"{existing_dim} dimensions but this batch has {embedding_dim}. "
                "Clear the vector store or re-embed with a consistent model/dimension setting."
            )

    # --- Document metadata tracking ---

    def _load_embedded_documents_locked(self) -> dict[str, int]:
        if self._embedded_docs_cache is not None:
            return dict(self._embedded_docs_cache)

        meta_path = self.persist_directory / "embedded_docs.json"
        docs: dict[str, int] = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    content = f.read().strip()
                    if content:
                        docs = json.loads(content)
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

        self._embedded_docs_cache = {
            str(key): int(value) for key, value in docs.items()
        }
        return dict(self._embedded_docs_cache)

    def _write_embedded_documents_locked(self, docs: dict[str, int]) -> None:
        meta_path = self.persist_directory / "embedded_docs.json"
        normalized = {str(key): int(value) for key, value in docs.items()}
        tmp_path = meta_path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(normalized, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, meta_path)
            self._embedded_docs_cache = dict(normalized)
        except IOError as e:
            logger.error("Failed to save embedded_docs.json: %s", e)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

    def _document_rows_exist_locked(self, document_key: str) -> bool:
        if not document_key or not self.sentences_table:
            return False
        try:
            rows = self._safe_select(
                self.sentences_table.search()
                .where(self._where_eq("document_key", document_key))
                .limit(1),
                ["id"],
            ).to_list()
            return bool(rows)
        except Exception:
            return False

    def get_embedded_documents(self) -> dict[str, int]:
        """Return dict mapping document keys to their sentence count."""
        with self._embedded_docs_lock:
            return self._load_embedded_documents_locked()

    def save_embedded_documents(self, docs: dict[str, int], allow_empty: bool = True):
        if not docs and not allow_empty:
            return

        with self._embedded_docs_lock:
            self._write_embedded_documents_locked(docs)

    def update_embedded_document(self, document_key: str, sentence_count: int) -> None:
        with self._embedded_docs_lock:
            docs = self._load_embedded_documents_locked()
            docs[document_key] = sentence_count
            self._write_embedded_documents_locked(docs)

    # --- Sentence operations ---

    def add_sentences(
        self,
        sentences: List[Sentence],
        embeddings: List[List[float]],
        document_key: str,
        batch_size: int = 5000,
    ):
        """Add sentence embeddings to the store."""
        if not sentences or not embeddings:
            return

        embedding_dim = self._validate_batch_dimensions(embeddings)
        self._ensure_sentence_table_compatible(embedding_dim)

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
                rows.append(
                    self._normalize_sentence_row(
                        {
                            "id": str(s.id),
                            "vector": [float(x) for x in emb],
                            "document": str(s.text),
                            "page": int(s.page),
                            "page_section": int(s.page_section)
                            if s.page_section is not None
                            else None,
                            "sentence_index": int(s.sentence_index),
                            "citation_numbers": _clip_list(
                                [int(num) for num in (s.citation_numbers or [])]
                            ),
                            "referenced_texts": _clip_list(
                                [str(t) for t in (s.referenced_texts or [])], limit=20
                            ),
                            "referenced_bibtex": _clip_list(
                                [str(b) for b in (s.referenced_bibtex or [])], limit=20
                            ),
                        },
                        document_key=document_key,
                    )
                )

            with self._table_lock:
                if self.sentences_table is None:
                    self.sentences_table = self.db.create_table(
                        self._TABLE_NAME,
                        data=self._rows_to_arrow_table(rows, embedding_dim),
                        mode="overwrite",
                    )
                    self._detected_sentence_dim = embedding_dim
                else:
                    self.sentences_table.add(rows)

        self._ensure_cosine_index()

    def get_sentences(self, document_key: str) -> List[Sentence]:
        if not self._prepare_for_read():
            return []

        try:
            rows = self._safe_select(
                self.sentences_table.search().where(
                    self._where_eq("document_key", document_key)
                ),
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
            ).to_list()
        except Exception:
            return []

        rows.sort(
            key=lambda r: (
                int(r.get("page", 1) or 1),
                int(r.get("sentence_index", 0) or 0),
            )
        )

        out: list[Sentence] = []
        for row in rows:
            out.append(
                Sentence(
                    id=str(row.get("id", "")),
                    document_id=str(row.get("document_key", document_key)),
                    page=int(row.get("page", 1) or 1),
                    page_section=(
                        int(row["page_section"])
                        if row.get("page_section") is not None
                        else None
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
        if not ids or not self._prepare_for_read():
            return {}

        out: dict[str, dict] = {}
        for sid in [str(i) for i in ids]:
            try:
                row_list = self._safe_select(
                    self.sentences_table.search()
                    .where(self._where_eq("id", sid))
                    .limit(1),
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
                ).to_list()
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
        if not self._prepare_for_read():
            return [], [], []

        def _execute_query() -> list[dict]:
            builder: Any = self.sentences_table.search(
                [float(x) for x in query_embedding]
            )
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

        try:
            rows = _execute_query()
        except Exception:
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
            score = 1.0 - (distance / 2)
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
        if not ids or not self._prepare_for_read():
            return {}

        out: dict[str, str] = {}
        for sid in [str(i) for i in ids]:
            try:
                row_list = self._safe_select(
                    self.sentences_table.search()
                    .where(self._where_eq("id", sid))
                    .limit(1),
                    ["id", "document"],
                ).to_list()
            except Exception:
                continue
            if not row_list:
                continue
            doc = row_list[0].get("document")
            if doc is not None:
                out[sid] = str(doc)
        return out

    def is_document_embedded(self, document_key: str) -> bool:
        if not document_key or not self._prepare_for_read():
            return False

        try:
            rows = self._safe_select(
                self.sentences_table.search()
                .where(self._where_eq("document_key", document_key))
                .limit(1),
                ["id"],
            ).to_list()
            return bool(rows)
        except Exception:
            return False

    # --- Cleanup ---

    def delete_document(self, document_key: str):
        if not self.sentences_table:
            return
        try:
            self.sentences_table.delete(self._where_eq("document_key", document_key))
            with self._embedded_docs_lock:
                docs = self._load_embedded_documents_locked()
                if document_key in docs:
                    del docs[document_key]
                    self._write_embedded_documents_locked(docs)
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

        with self._embedded_docs_lock:
            self._embedded_docs_cache = {}
            meta_path = self.persist_directory / "embedded_docs.json"
            try:
                if meta_path.exists():
                    meta_path.unlink()
            except OSError:
                logger.debug("Failed to remove embedded_docs.json during clear_all")

    def get_sentence_count(self) -> int:
        if not self._prepare_for_read():
            return 0
        try:
            return int(self.sentences_table.count_rows())
        except Exception:
            return 0

    def get_document_title(self, document_key: str) -> Optional[str]:
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
