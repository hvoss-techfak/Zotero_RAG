"""Background embedding manager with ThreadPoolExecutor."""

import logging
import os
import random
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Callable

import ollama
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from zoterorag.config import Config
from zoterorag.models import Document, Sentence, EmbeddingStatus
from zoterorag.pdf_processor import PDFProcessor
from zoterorag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages sentence embedding with background processing."""

    def __init__(self, config: Config, zotero_client=None):
        self.config = config
        self.vector_store = VectorStore(str(config.VECTOR_STORE_DIR))
        self.pdf_processor = PDFProcessor()
        self._executor: Optional[ThreadPoolExecutor] = None
        self.zotero_client = zotero_client

        self._embedding_progress: EmbeddingStatus = EmbeddingStatus()
        self._progress_lock = threading.Lock()

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _status_copy_locked(self) -> EmbeddingStatus:
        return EmbeddingStatus(
            total_documents=self._embedding_progress.total_documents,
            processed_documents=self._embedding_progress.processed_documents,
            embedded_sections=self._embedding_progress.embedded_sections,
            embedded_sentences=self._embedding_progress.embedded_sentences,
            pending_sections=self._embedding_progress.pending_sections,
            is_running=self._embedding_progress.is_running,
            failed_documents=self._embedding_progress.failed_documents,
            started_at=self._embedding_progress.started_at,
            finished_at=self._embedding_progress.finished_at,
            last_error=self._embedding_progress.last_error,
        )

    def get_embedding_status(self) -> EmbeddingStatus:
        with self._progress_lock:
            return self._status_copy_locked()

    def _update_progress(self, status: EmbeddingStatus):
        with self._progress_lock:
            if status.total_documents > 0:
                self._embedding_progress.total_documents = status.total_documents

            if status.processed_documents > 0:
                self._embedding_progress.processed_documents = max(
                    self._embedding_progress.processed_documents,
                    status.processed_documents,
                )

            self._embedding_progress.embedded_sections += max(
                0, status.embedded_sections
            )
            self._embedding_progress.embedded_sentences += max(
                0, status.embedded_sentences
            )
            self._embedding_progress.pending_sections = max(0, status.pending_sections)
            self._embedding_progress.failed_documents += max(0, status.failed_documents)
            self._embedding_progress.is_running = status.is_running

            if status.started_at:
                self._embedding_progress.started_at = status.started_at
            if status.finished_at:
                self._embedding_progress.finished_at = status.finished_at
            if status.last_error:
                self._embedding_progress.last_error = status.last_error

    def mark_document_completed(
        self,
        *,
        embedded_sections: int = 0,
        embedded_sentences: int = 0,
        failed: bool = False,
        last_error: str = "",
    ) -> EmbeddingStatus:
        with self._progress_lock:
            self._embedding_progress.processed_documents += 1
            self._embedding_progress.embedded_sections += max(0, embedded_sections)
            self._embedding_progress.embedded_sentences += max(0, embedded_sentences)
            self._embedding_progress.is_running = True
            if failed:
                self._embedding_progress.failed_documents += 1
            if last_error:
                self._embedding_progress.last_error = last_error
            return self._status_copy_locked()

    def start_embedding_job(self, total_documents: int):
        with self._progress_lock:
            self._embedding_progress = EmbeddingStatus(
                total_documents=total_documents,
                processed_documents=0,
                is_running=True,
                started_at=self._utc_now_iso(),
                finished_at="",
                last_error="",
                failed_documents=0,
            )
        logger.info(
            "[EmbeddingManager] Started embedding job for %s documents",
            total_documents,
        )

    def finish_embedding_job(self, *, last_error: str = "") -> EmbeddingStatus:
        with self._progress_lock:
            self._embedding_progress.is_running = False
            self._embedding_progress.finished_at = self._utc_now_iso()
            if last_error:
                self._embedding_progress.last_error = last_error
            return self._status_copy_locked()

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

    def shutdown(self):
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _embed_text(self, texts: List[str]) -> List[List[float]]:
        logger.debug(f"Embedding single text (batch size 1): {texts[0][:60]}...")
        response = ollama.embeddings(
            model=self.config.EMBEDDING_MODEL,
            prompt=texts[0],
            options=self._get_embedding_options(),
        )
        return [response["embedding"]]

    def embed_text(self, text_list: List[str]) -> List[List[float]]:
        # we need to request manually
        # convert list to the format expected by ollama json encoding of list string
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

        # to json
        response = response.json()
        embeddings = response.get("embeddings")
        if (
            not embeddings
            or not isinstance(embeddings, list)
            or not all(isinstance(e, list) for e in embeddings)
        ):
            raise ValueError(f"Unexpected embedding response format: {response}")

        return embeddings

    def _embed_batch_ollama(self, texts: List[str]) -> List[List[float]]:
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
        logger.info(
            f"Embedding batch of {len(normalized)} texts with batch size {batch_size} and embedding dimensions {self.config.EMBEDDING_DIMENSIONS}"
        )
        logger.debug(
            "Sample text for embedding: %s",
            random.choice(normalized) + "..." if normalized else "No texts",
        )
        all_embeddings: List[List[float]] = []

        for i in range(0, len(normalized), batch_size):
            chunk = normalized[i : i + batch_size]
            all_embeddings.extend(self.embed_text(chunk))

        return all_embeddings

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._embed_batch_ollama(texts)

    # --- Sentence extraction ---

    def process_document(
        self,
        document: Document,
        pdf_path: str,
    ) -> List[Sentence]:
        return self.pdf_processor.extract_sentences(
            pdf_path, document_id=document.zotero_key
        )

    # --- Background operations ---

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
        try:
            # Prefer DB-backed check: if any sentence vectors exist, it's embedded.
            if self.vector_store.is_document_embedded(document.zotero_key):
                logger.info(
                    "[Embedding] Skipping already-embedded document %s (%s)",
                    (document.title or "")[:60],
                    document.zotero_key,
                )

                snapshot = self.mark_document_completed()
                if callback:
                    callback(snapshot)

                f: Future = Future()
                f.set_result(None)
                return f

            # Fall back to metadata file check (kept for compatibility)
            embedded = self.vector_store.get_embedded_documents()
            if embedded.get(document.zotero_key, 0) > 0:
                logger.info(
                    "[Embedding] Skipping already-embedded document %s (%s)",
                    (document.title or "")[:60],
                    document.zotero_key,
                )

                snapshot = self.mark_document_completed()
                if callback:
                    callback(snapshot)

                f: Future = Future()
                f.set_result(None)
                return f
        except Exception as e:
            logger.debug(
                "[Embedding] Skip-check failed for %s: %s", document.zotero_key, e
            )

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

            sentences = self.process_document(document, pdf_path)

            sent_embeddings = self.embed_batch([s.text for s in sentences])
            self.vector_store.add_sentences(
                sentences, sent_embeddings, document_key=doc_key
            )
            self.vector_store.update_embedded_document(doc_key, len(sentences))

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
        """Scan a directory for PDF files and create Document objects.

        This helper is used by tests and the optional CLI.
        """
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
        """Calculate relevance score from embedding vector.

        Kept as a small utility used by tests and optional reranking approaches.
        """
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
    from zoterorag.logging_setup import setup_logging

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

    # Ensure directories exist
    Config.ensure_dirs()

    # Initialize embedding manager
    config = Config()
    manager = EmbeddingManager(config)

    # Get documents from PDF directory
    pdf_dir = Path(args.pdf_dir)
    documents = manager.get_pdf_documents_from_directory(pdf_dir)

    if not documents:
        print(f"No PDFs found in {pdf_dir}")
        sys.exit(1)

    # Filter to unembedded docs
    embedded = manager.vector_store.get_embedded_documents()
    pending = [
        (doc, path)
        for doc, path in documents
        if doc.zotero_key not in embedded or embedded[doc.zotero_key] == 0
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
