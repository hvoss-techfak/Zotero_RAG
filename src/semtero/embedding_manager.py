"""Background embedding manager with ThreadPoolExecutor."""

import logging
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Callable

import requests

from semtero.config import Config
from semtero.models import Document, Sentence, EmbeddingStatus
from semtero.pdf_processor import PDFProcessor
from semtero.vector_store import VectorStore

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages sentence embedding with background processing."""

    def __init__(
        self,
        config: Config,
        zotero_client=None,
        vector_store: VectorStore | None = None,
    ):
        self.config = config
        self.vector_store = vector_store or VectorStore(str(config.VECTOR_STORE_DIR))
        self.pdf_processor = PDFProcessor()
        self._executor: Optional[ThreadPoolExecutor] = None
        self.zotero_client = zotero_client

        self._status = EmbeddingStatus()
        self._lock = threading.Lock()
        self._ollama_lock = threading.Lock()
        self._ollama_paused = threading.Event()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _snapshot(self) -> EmbeddingStatus:
        s = self._status
        return EmbeddingStatus(
            total_documents=s.total_documents,
            processed_documents=s.processed_documents,
            embedded_sections=s.embedded_sections,
            embedded_sentences=s.embedded_sentences,
            pending_sections=s.pending_sections,
            is_running=s.is_running,
            failed_documents=s.failed_documents,
            started_at=s.started_at,
            finished_at=s.finished_at,
            last_error=s.last_error,
        )

    def _has_record(self, document_key: str) -> bool:
        if not document_key:
            return False
        try:
            embedded = self.vector_store.get_embedded_documents()
        except Exception:
            return False
        return document_key in embedded

    def _mark_zero_and_notify(
        self,
        doc_key: str,
        title: str,
        callback: Optional[Callable[[EmbeddingStatus], None]] = None,
    ) -> EmbeddingStatus:
        self.vector_store.update_embedded_document(doc_key, 0)
        logger.info(
            "[Embedding] No extractable sentences for %s (%s); marking as processed.",
            (title or "")[:60],
            doc_key,
        )
        snapshot = self.mark_document_completed()
        if callback:
            callback(snapshot)
        return snapshot

    def get_embedding_status(self) -> EmbeddingStatus:
        with self._lock:
            return self._snapshot()

    # --- Progress tracking (shared between sync & remote embedding paths) ---

    def _update_progress(self, status: EmbeddingStatus):
        with self._lock:
            p = self._status
            if status.total_documents > 0:
                p.total_documents = status.total_documents
            if status.processed_documents > 0:
                p.processed_documents = max(p.processed_documents, status.processed_documents)
            p.embedded_sections += max(0, status.embedded_sections)
            p.embedded_sentences += max(0, status.embedded_sentences)
            p.pending_sections = max(0, status.pending_sections)
            p.failed_documents += max(0, status.failed_documents)
            p.is_running = status.is_running
            if status.started_at:
                p.started_at = status.started_at
            if status.finished_at:
                p.finished_at = status.finished_at
            if status.last_error:
                p.last_error = status.last_error

    def _update(
        self,
        *,
        processed: int = 0,
        sections: int = 0,
        sentences: int = 0,
        failed: bool = False,
        error: str = "",
    ) -> EmbeddingStatus:
        with self._lock:
            self._status.processed_documents += processed
            self._status.embedded_sections += max(0, sections)
            self._status.embedded_sentences += max(0, sentences)
            self._status.is_running = True
            if failed:
                self._status.failed_documents += 1
            if error:
                self._status.last_error = error
            return self._snapshot()

    def mark_document_completed(
        self,
        *,
        embedded_sections: int = 0,
        embedded_sentences: int = 0,
        failed: bool = False,
        last_error: str = "",
    ) -> EmbeddingStatus:
        return self._update(
            processed=1,
            sections=embedded_sections,
            sentences=embedded_sentences,
            failed=failed,
            error=last_error,
        )

    def mark_embedding_scan_started(self) -> EmbeddingStatus:
        with self._lock:
            self._status = EmbeddingStatus(
                total_documents=0,
                processed_documents=0,
                embedded_sections=0,
                embedded_sentences=0,
                pending_sections=0,
                is_running=True,
                failed_documents=0,
                started_at=self._now(),
                finished_at="",
                last_error="",
            )
            return self._snapshot()

    def set_embedding_job_total(self, total_documents: int) -> EmbeddingStatus:
        with self._lock:
            self._status.total_documents = max(
                int(total_documents),
                self._status.processed_documents,
                self._status.total_documents,
            )
            self._status.is_running = True
            if not self._status.started_at:
                self._status.started_at = self._now()
            self._status.finished_at = ""
            return self._snapshot()

    def start_embedding_job(self, total_documents: int) -> EmbeddingStatus:
        snapshot = self.mark_embedding_scan_started()
        if total_documents > 0:
            snapshot = self.set_embedding_job_total(total_documents)
        logger.info(
            "[EmbeddingManager] Started embedding job for %s documents", total_documents
        )
        return snapshot

    def finish_embedding_job(self, *, last_error: str = "") -> EmbeddingStatus:
        with self._lock:
            self._status.is_running = False
            self._status.finished_at = self._now()
            if last_error:
                self._status.last_error = last_error
            return self._snapshot()

    @property
    def executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            max_workers = getattr(
                self.config,
                "MAX_EMBEDDING_WORKERS",
                max(1, (os.cpu_count() or 2) // 2),
            )
            logger.info(
                "[EmbeddingManager] Initializing ThreadPoolExecutor with %s workers",
                max_workers,
            )
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
        return self._executor

    def pause_ollama(self) -> None:
        """Pause background Ollama embedding calls so a search can jump ahead."""
        self._ollama_paused.set()

    def resume_ollama(self) -> None:
        """Resume background Ollama embedding calls after a search finishes."""
        self._ollama_paused.clear()

    def shutdown(self):
        self.resume_ollama()
        if self._executor:
            logger.info("[EmbeddingManager] Shutting down ThreadPoolExecutor")
            self._executor.shutdown(wait=True)
            self._executor = None

    # --- Embedding generation ---

    def _get_embedding_options(self) -> dict:
        opts = {"num_ctx": 32768}
        if self.config.EMBEDDING_DIMENSIONS > 0:
            opts["dimensions"] = self.config.EMBEDDING_DIMENSIONS
        return opts

    def _get_store_dimension(self) -> int | None:
        try:
            detected = self.vector_store.get_detected_dimension()
        except Exception:
            return None
        return detected if isinstance(detected, int) and detected > 0 else None

    def _validate_embeddings(
        self, embeddings: List[List[float]], *, context: str = "embedding request"
    ) -> List[List[float]]:
        if not embeddings:
            raise ValueError(f"{context} returned empty embeddings")

        dims = sorted({len(emb or []) for emb in embeddings})
        if not dims or dims == [0]:
            raise ValueError(f"{context} returned empty embeddings")
        if len(dims) != 1:
            raise ValueError(
                f"{context} returned inconsistent embedding dimensions: {dims}"
            )

        actual_dim = dims[0]
        expected_dim = int(getattr(self.config, "EMBEDDING_DIMENSIONS", 0) or 0)
        if expected_dim > 0 and actual_dim != expected_dim:
            raise ValueError(
                "Embedding provider returned "
                f"{actual_dim}-dimensional vectors but EMBEDDING_DIMENSIONS={expected_dim}. "
                "This usually means the model/server ignored the requested dimensions."
            )

        store_dim = self._get_store_dimension()
        if store_dim and actual_dim != store_dim:
            raise ValueError(
                "Embedding provider returned "
                f"{actual_dim}-dimensional vectors but the existing vector store uses {store_dim}. "
                "Clear the vector store or re-embed with a consistent model/dimension setting."
            )

        return [[float(x) for x in emb] for emb in embeddings]

    def _ollama_embed(self, text_list: List[str]) -> List[List[float]]:
        response = requests.post(
            url=f"{self.config.OLLAMA_BASE_URL}/api/embed",
            json={
                "model": self.config.EMBEDDING_MODEL,
                "input": text_list,
                "options": self._get_embedding_options(),
            },
        )
        if response.status_code != 200:
            raise ValueError(
                f"Embedding request failed with status {response.status_code}: {response.text}"
            )

        response = response.json()
        embeddings = response.get("embeddings")
        if (
            not embeddings
            or not isinstance(embeddings, list)
            or not all(isinstance(e, list) for e in embeddings)
        ):
            raise ValueError(f"Unexpected embedding response format: {response}")

        return self._validate_embeddings(embeddings, context="batch embedding request")

    def embed_text(self, text_list: List[str]) -> List[List[float]]:
        while self._ollama_paused.is_set():
            time.sleep(0.05)
        with self._ollama_lock:
            return self._ollama_embed(text_list)

    def embed_text_priority(self, text_list: List[str]) -> List[List[float]]:
        with self._ollama_lock:
            return self._ollama_embed(text_list)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        normalized: List[str] = []
        for t in texts:
            if t is None:
                normalized.append("")
            elif isinstance(t, str):
                normalized.append(t)
            else:
                normalized.append(str(t))

        batch_size = getattr(self.config, "BATCH_EMBEDDING_SIZE", 32)
        all_embeddings: List[List[float]] = []

        for i in range(0, len(normalized), batch_size):
            chunk = normalized[i : i + batch_size]
            all_embeddings.extend(self.embed_text(chunk))

        return all_embeddings

    # --- Sentence extraction ---

    def process_document(self, document: Document, pdf_path: str) -> List[Sentence]:
        return self.pdf_processor.extract_sentences(
            pdf_path, document_id=document.zotero_key
        )

    # --- Background operations ---

    def _embed_and_store(
        self, document: Document, pdf_path: str
    ) -> list[Sentence] | None:
        """Extract sentences, embed, and store. Returns sentences on success, None if empty."""
        sentences = self.process_document(document, pdf_path)
        if not sentences:
            return None

        sent_embeddings = self.embed_batch([s.text for s in sentences])
        self.vector_store.add_sentences(
            sentences, sent_embeddings, document_key=document.zotero_key
        )
        self.vector_store.update_embedded_document(
            document.zotero_key, len(sentences)
        )
        return sentences

    def embed_document_async(
        self,
        document: Document,
        pdf_path: str,
        callback: Optional[Callable[[EmbeddingStatus], None]] = None,
    ) -> Future:
        return self.executor.submit(
            self._embed_document_task, document, pdf_path, callback
        )

    def embed_document_async_with_client(
        self,
        document: Document,
        zotero_client,
        callback: Optional[Callable[[EmbeddingStatus], None]] = None,
    ) -> Future:
        key = document.zotero_key
        try:
            if self.vector_store.is_document_embedded(key):
                logger.info(
                    "[Embedding] Skipping already-embedded document %s (%s)",
                    (document.title or "")[:60],
                    key,
                )
                snapshot = self.mark_document_completed()
                if callback:
                    callback(snapshot)
                f: Future = Future()
                f.set_result(None)
                return f

            if self._has_record(key):
                logger.info(
                    "[Embedding] Skipping already-processed document %s (%s)",
                    (document.title or "")[:60],
                    key,
                )
                snapshot = self.mark_document_completed()
                if callback:
                    callback(snapshot)
                f: Future = Future()
                f.set_result(None)
                return f
        except Exception as e:
            logger.debug("[Embedding] Skip-check failed for %s: %s", key, e)

        return self.executor.submit(
            self._embed_document_from_zotero_task, document, zotero_client, callback
        )

    def _embed_document_from_zotero_task(
        self,
        document: Document,
        zotero_client,
        callback: Optional[Callable[[EmbeddingStatus], None]],
    ):
        doc_key = document.zotero_key
        start_time = time.time()

        if document.group_id is not None:
            pdf_bytes = zotero_client.get_group_pdf_bytes(document.group_id, doc_key)
        else:
            pdf_bytes = zotero_client.get_pdf_bytes(doc_key)

        if pdf_bytes is None:
            snapshot = self.mark_document_completed(
                failed=True,
                last_error=f"No PDF available for document {doc_key}",
            )
            if callback:
                callback(snapshot)
            return

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                temp_path = Path(tmp.name)

            sentences = self.process_document(document, str(temp_path))
            if not sentences:
                self._mark_zero_and_notify(doc_key, document.title, callback)
                return

            sent_embeddings = self.embed_batch([s.text for s in sentences])
            try:
                self.vector_store.add_sentences(
                    sentences, sent_embeddings, document_key=doc_key
                )
                self.vector_store.update_embedded_document(doc_key, len(sentences))
            except Exception as e:
                logger.error(
                    "Failed to store embeddings for document %s: %s", doc_key, e
                )
                snapshot = self.mark_document_completed(
                    failed=True,
                    last_error=f"Failed to store embeddings for {doc_key}: {e}",
                )
                if callback:
                    callback(snapshot)
                return

            snapshot = self.mark_document_completed(embedded_sentences=len(sentences))
            if callback:
                callback(snapshot)

            elapsed = time.time() - start_time
            logger.debug(
                "[Embedding] Complete: %s (%s) in %.1fs - %s sentences",
                document.title[:40],
                doc_key,
                elapsed,
                len(sentences),
            )

        except Exception as e:
            logger.error("Failed to embed document %s: %s", document.zotero_key, e)
            snapshot = self.mark_document_completed(
                failed=True,
                last_error=f"Failed to embed document {document.zotero_key}: {e}",
            )
            if callback:
                callback(snapshot)
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def _embed_document_task(
        self,
        document: Document,
        pdf_path: str,
        callback: Optional[Callable[[EmbeddingStatus], None]],
    ):
        doc_key = document.zotero_key
        start_time = time.time()

        try:
            if callback:
                callback(EmbeddingStatus(is_running=True, pending_sections=1))

            sentences = self._embed_and_store(document, pdf_path)
            if sentences is None:
                self._mark_zero_and_notify(doc_key, document.title, callback)
                return

            snapshot = self.mark_document_completed(embedded_sentences=len(sentences))
            if callback:
                callback(snapshot)

            elapsed = time.time() - start_time
            logger.debug(
                "[Embedding] Complete: %s (%s) in %.1fs - %s sentences",
                document.title[:40],
                doc_key,
                elapsed,
                len(sentences),
            )

        except Exception as e:
            logger.error("Failed to embed document %s: %s", document.zotero_key, e)
            snapshot = self.mark_document_completed(
                failed=True,
                last_error=f"Failed to embed document {document.zotero_key}: {e}",
            )
            if callback:
                callback(snapshot)

    @classmethod
    def get_pdf_documents_from_directory(
        cls, pdf_dir: Path
    ) -> List[tuple[Document, str]]:
        documents: list[tuple[Document, str]] = []
        if not pdf_dir.exists():
            return documents

        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            key = pdf_path.stem
            doc = Document(
                zotero_key=key,
                title=pdf_path.name,
                pdf_path=pdf_path,
            )
            documents.append((doc, str(pdf_path)))

        return documents

    def calculate_relevance_score(self, embedding: List[float]) -> float:
        if not embedding:
            return 0.0

        sum_positive = 0.0
        sum_total = 0.0

        for val in embedding:
            sum_total += val * val
            if val > 0:
                sum_positive += val

        if sum_total == 0:
            return 0.0

        magnitude = (sum_total**0.5) / len(embedding)
        positive_ratio = sum_positive / len(embedding)
        return (magnitude + positive_ratio) / 2


if __name__ == "__main__":
    from semtero.logging_setup import setup_logging

    setup_logging(level=os.getenv("LOG_LEVEL", "WARNING"))

    import argparse

    parser = argparse.ArgumentParser(description="Run embedding without MCP server")
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="./data/pdfs",
        help="Directory containing PDF files to embed",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop on first error instead of continuing",
    )
    args = parser.parse_args()

    Config.ensure_dirs()

    config = Config()
    manager = EmbeddingManager(config)

    pdf_dir = Path(args.pdf_dir)
    documents = manager.get_pdf_documents_from_directory(pdf_dir)

    if not documents:
        print(f"No PDFs found in {pdf_dir}")
        sys.exit(1)

    pending = [
        (doc, path)
        for doc, path in documents
        if not manager._has_record(doc.zotero_key)
    ]

    print(f"Total PDFs: {len(documents)}, Pending: {len(pending)}")

    for doc, _ in pending:
        print(f"  Pending: {doc.title} ({doc.zotero_key})")
        pdf_processor = PDFProcessor()
        try:
            sentences = pdf_processor.extract_sentences(
                str(pdf_dir / f"{doc.zotero_key}.pdf")
            )
            print(f"    Extracted {len(sentences)} sentences")
            manager.embed_batch([s.text for s in sentences])
        except Exception as e:
            print(f"    Failed to extract sentences: {e}")
