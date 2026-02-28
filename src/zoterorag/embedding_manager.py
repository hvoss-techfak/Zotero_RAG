"""Background embedding manager with ThreadPoolExecutor."""

import argparse
import logging
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Optional, List, Callable

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Config
from .models import Document, Section, SentenceWindow, EmbeddingStatus
from .pdf_processor import PDFProcessor
from .vector_store import VectorStore


logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages hierarchical embedding with background processing."""

    def __init__(self, config: Config, zotero_client=None):
        self.config = config
        self.vector_store = VectorStore(config.VECTOR_STORE_DIR)
        # Use page splits from config for line-count based section extraction
        self.pdf_processor = PDFProcessor(page_splits=config.PAGE_SPLITS)
        self._executor: Optional[ThreadPoolExecutor] = None
        # Optional ZoteroClient for on-demand PDF fetching (temp file processing)
        self.zotero_client = zotero_client
        
        # Global progress tracking for background embedding jobs
        self._embedding_progress: EmbeddingStatus = EmbeddingStatus()
        self._progress_lock = threading.Lock()

    def get_embedding_status(self) -> EmbeddingStatus:
        """Get the current embedding status with thread-safe access."""
        with self._progress_lock:
            return EmbeddingStatus(
                total_documents=self._embedding_progress.total_documents,
                processed_documents=self._embedding_progress.processed_documents,
                embedded_sections=self._embedding_progress.embedded_sections,
                embedded_sentences=self._embedding_progress.embedded_sentences,
                pending_sections=self._embedding_progress.pending_sections,
                is_running=self._embedding_progress.is_running
            )

    def _update_progress(self, status: EmbeddingStatus):
        """Thread-safe update of embedding progress."""
        with self._progress_lock:
            if status.total_documents > 0:
                self._embedding_progress.total_documents = status.total_documents

            if status.processed_documents > 0:
                # Never allow processed_documents to go backwards.
                self._embedding_progress.processed_documents = max(
                    self._embedding_progress.processed_documents,
                    status.processed_documents,
                )

            self._embedding_progress.embedded_sections += status.embedded_sections
            self._embedding_progress.embedded_sentences += status.embedded_sentences

            # is_running is a state flag; allow it to flip either direction.
            self._embedding_progress.is_running = status.is_running

    def start_embedding_job(self, total_documents: int):
        """Start a new embedding job and initialize progress tracking.
        
        Args:
            total_documents: Total number of documents to embed in this job
        """
        with self._progress_lock:
            self._embedding_progress = EmbeddingStatus(
                total_documents=total_documents,
                processed_documents=0,
                is_running=True
            )
        logger.info(f"[EmbeddingManager] Started embedding job for {total_documents} documents")

    @property
    def executor(self) -> ThreadPoolExecutor:
        """Lazy initialization of thread pool."""
        if self._executor is None:
            max_workers = getattr(self.config, 'MAX_EMBEDDING_WORKERS', max(1,os.cpu_count()//2))
            logger.info(f"[EmbeddingManager] Initializing ThreadPoolExecutor with {max_workers} workers")
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
            logger.info(f"[EmbeddingManager] ThreadPoolExecutor initialized successfully")
        return self._executor

    def shutdown(self):
        """Shutdown the executor."""
        if self._executor:
            logger.info("[EmbeddingManager] Shutting down ThreadPoolExecutor")
            self._executor.shutdown(wait=True)
            self._executor = None

    # --- Embedding generation ---

    def _get_embedding_options(self) -> dict:
        """Get Ollama options for embedding, using config dimensions if set."""
        opts = {"num_ctx": 32768}
        if self.config.EMBEDDING_DIMENSIONS > 0:
            opts["dimensions"] = self.config.EMBEDDING_DIMENSIONS
        return opts

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _embed_text(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama."""
        response = ollama.embeddings(
            model=self.config.EMBEDDING_MODEL,
            prompt=texts[0],  # Single text at a time for accuracy
            options=self._get_embedding_options()
        )
        return [response["embedding"]]

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        response = ollama.embeddings(
            model=self.config.EMBEDDING_MODEL,
            prompt=text,
            options=self._get_embedding_options()
        )
        return response["embedding"]

    def _embed_batch_ollama(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using Ollama.

        Note: The `ollama` Python client versions differ in whether they accept a
        list for `prompt`. Some versions validate `prompt` as a single string
        (Pydantic `EmbeddingsRequest.prompt: str`) and will raise a validation
        error if a list is passed.

        To stay compatible and avoid hard dependency pinning, we embed in chunks
        but send one request per text (still benefiting from our own chunking
        and retry logic).

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors, one per input text
        """
        if not texts:
            return []

        # Normalize defensively: downstream should always see strings.
        normalized: List[str] = []
        for t in texts:
            if t is None:
                normalized.append("")
            elif isinstance(t, str):
                normalized.append(t)
            else:
                normalized.append(str(t))

        batch_size = getattr(self.config, 'BATCH_EMBEDDING_SIZE', 32)
        all_embeddings: List[List[float]] = []

        for i in range(0, len(normalized), batch_size):
            chunk = normalized[i:i + batch_size]

            # Embed each text individually to satisfy `prompt: str`.
            for text in chunk:
                try:
                    all_embeddings.append(self.embed_text(text))
                except Exception as e:
                    # Preserve existing behaviour: surface failures but keep going.
                    logger.warning(
                        f"Embedding failed for item {i + len(all_embeddings) - i} in chunk {i//batch_size}: {e}"
                    )
                    all_embeddings.append([])

        return all_embeddings

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        return self._embed_batch_ollama(texts)

    def calculate_relevance_score(self, embedding: List[float]) -> float:
        """Calculate relevance score from embedding vector.
        
        Uses the cross-encoder approach where higher magnitude and positive values
        in the embedding indicate stronger relevance to the query.
        
        Args:
            embedding: The embedding vector from reranker model
            
        Returns:
            Relevance score between 0 and 1
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
        
        # Normalize and combine magnitude with positive bias
        magnitude = (sum_total ** 0.5) / len(embedding)
        positive_ratio = sum_positive / len(embedding)
        
        return (magnitude + positive_ratio) / 2

    def process_document(
        self,
        document: Document,
        pdf_path: str | None = None,
        embed_sentences: bool = False
    ) -> tuple[List[Section], List[SentenceWindow]]:
        """Extract sections and optionally sentence windows from PDF.
        
        Uses line-count based splitting (page_splits config) instead of heading-based.
        
        Args:
            document: The document to process
            pdf_path: Path to the PDF file. If None, will fetch from zotero_client
            embed_sentences: Whether to create sentence windows
            
        Returns:
            Tuple of (sections, sentence_windows)
        """
        # Use extract_quarter_sections for line-count based splitting
        sections = self.pdf_processor.extract_quarter_sections(pdf_path)

        # Link sections to parent for hierarchy
        self._link_sections(sections)

        sentence_windows: List[SentenceWindow] = []
        if embed_sentences:
            for section in sections:
                windows = self.pdf_processor.create_sentence_windows(section)
                sentence_windows.extend(windows)

        return sections, sentence_windows

    def _link_sections(self, sections: List[Section]):
        """Build parent-child relationships between sections."""
        # Note: Section model doesn't have parent_id - this is for reference only
        pass  # Parent linking handled via document hierarchy in search

    def embed_sections(self, sections: List[Section]) -> tuple[List[str], List[List[float]]]:
        """Generate embeddings for all sections."""
        texts = [s.text for s in sections]
        return self.embed_batch(texts), list(range(len(sections)))

    def embed_sentence_windows(
        self,
        windows: List[SentenceWindow]
    ) -> tuple[List[str], List[List[float]]]:
        """Generate embeddings for all sentence windows."""
        texts = [w.text for w in windows]
        return self.embed_batch(texts), list(range(len(windows)))

    # --- Background operations ---

    def embed_document_async(
        self,
        document: Document,
        pdf_path: str,
        callback: Optional[Callable[[EmbeddingStatus], None]] = None
    ) -> Future:
        """Submit document for background embedding."""
        return self.executor.submit(
            self._embed_document_task,
            document, pdf_path, callback
        )

    def embed_document_async_with_client(
        self,
        document: Document,
        zotero_client,
        callback: Optional[Callable[[EmbeddingStatus], None]] = None
    ) -> Future:
        """Submit document for background embedding, fetching PDF on-demand.

        Args:
            document: The document to embed
            zotero_client: ZoteroClient instance to fetch PDF from
            callback: Optional progress callback

        Returns:
            Future object representing the async task
        """
        # Idempotency guard: don't re-embed already embedded docs.
        try:
            embedded = self.vector_store.get_embedded_documents()
            if embedded.get(document.zotero_key, 0) > 0:
                logger.info(
                    "[Embedding] Skipping already-embedded document %s (%s)",
                    document.title[:60],
                    document.zotero_key,
                )

                # Even if we skip the work, count this document as processed for the
                # current embedding job so progress reaches 100%.
                self._update_progress(
                    EmbeddingStatus(
                        processed_documents=self.get_embedding_status().processed_documents + 1,
                        is_running=True,
                    )
                )

                if callback:
                    # Emit a no-op progress update so callers waiting on activity
                    # still see that the task completed.
                    callback(EmbeddingStatus(is_running=True))

                f: Future = Future()
                f.set_result(None)
                return f
        except Exception as e:
            # If metadata is unreadable, fall back to embedding rather than risking
            # skipping work incorrectly.
            logger.debug("[Embedding] Skip-check failed for %s: %s", document.zotero_key, e)

        return self.executor.submit(
            self._embed_document_from_zotero_task,
            document, zotero_client, callback
        )

    def _embed_document_from_zotero_task(
        self,
        document: Document,
        zotero_client,
        callback: Optional[Callable[[EmbeddingStatus], None]]
    ):
        """Background task to fetch PDF from Zotero and embed it using temp file."""
        doc_key = document.zotero_key
        start_time = time.time()
        
        # Get PDF bytes from Zotero
        if document.group_id is not None:
            pdf_bytes = zotero_client.get_group_pdf_bytes(document.group_id, doc_key)
        else:
            pdf_bytes = zotero_client.get_pdf_bytes(doc_key)
        
        if pdf_bytes is None:
            #logger.error(f"Failed to fetch PDF for {doc_key} from Zotero, file was group file {document.group_id is not None}")
            if callback:
                status = EmbeddingStatus(is_running=False)
                callback(status)
            return
        
        # Create a temporary file, process it, then delete
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                temp_path = Path(tmp.name)
            
            logger.debug(f"[Embedding] Starting: {document.title[:50]}... ({doc_key})")

            # Extract sections (always do this)
            sections, sentence_windows = self.process_document(
                document, str(temp_path),
                embed_sentences=self.config.AUTO_EMBED_SENTENCES
            )
            
            logger.debug(f"[Embedding] Extracted {len(sections)} sections from {document.title[:30]}...")

            # Embed sections
            section_embeddings = self.embed_batch([s.text for s in sections])
            self.vector_store.add_sections(
                sections,
                section_embeddings,
                document_key=document.zotero_key,
                zotero_key=document.zotero_key,
                parent_item_key=document.parent_item_key,
            )

            logger.debug(f"[Embedding] Embedded {len(sections)} sections")

            # Update metadata (atomic)
            self.vector_store.update_embedded_document(document.zotero_key, len(sections))

            if callback:
                status = EmbeddingStatus(
                    embedded_sections=len(sections),
                    is_running=True
                )
                callback(status)

            # Optionally embed sentences in background too
            if sentence_windows and self.config.AUTO_EMBED_SENTENCES:
                logger.debug(f"[Embedding] Processing {len(sentence_windows)} sentence windows...")

                sent_embeddings = self.embed_batch([w.text for w in sentence_windows])
                self.vector_store.add_sentence_windows(
                    sentence_windows,
                    sent_embeddings,
                    document_key=document.zotero_key,
                    zotero_key=document.zotero_key,
                    parent_item_key=document.parent_item_key,
                )

            # ...existing code...

            elapsed = time.time() - start_time
            logger.debug(
                f"[Embedding] Complete: {document.title[:40]}... ({doc_key}) in {elapsed:.1f}s - "
                f"{len(sections)} sections, {len(sentence_windows) if sentence_windows else 0} sentences"
            )

        except Exception as e:
            logger.error(f"Failed to embed document {document.zotero_key}: {e}")
            if callback:
                status = EmbeddingStatus(is_running=False)
                callback(status)
        finally:
            # Always clean up temp file
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def _embed_document_task(
        self,
        document: Document,
        pdf_path: str,
        callback: Optional[Callable[[EmbeddingStatus], None]]
    ):
        """Background task to embed a single document."""
        doc_key = document.zotero_key
        start_time = time.time()
        
        try:
            if callback:
                status = EmbeddingStatus(is_running=True, pending_sections=1)
                callback(status)

            logger.debug(f"[Embedding] Starting: {document.title[:50]}... ({doc_key})")

            # Extract sections (always do this)
            sections, sentence_windows = self.process_document(
                document, pdf_path,
                embed_sentences=self.config.AUTO_EMBED_SENTENCES
            )
            
            logger.debug(f"[Embedding] Extracted {len(sections)} sections from {document.title[:30]}...")

            # Embed sections
            section_embeddings = self.embed_batch([s.text for s in sections])
            self.vector_store.add_sections(
                sections,
                section_embeddings,
                document_key=document.zotero_key,
                zotero_key=document.zotero_key,
                parent_item_key=document.parent_item_key,
            )

            logger.debug(f"[Embedding] Embedded {len(sections)} sections")

            # Update metadata (atomic)
            self.vector_store.update_embedded_document(document.zotero_key, len(sections))

            if callback:
                status = EmbeddingStatus(
                    embedded_sections=len(sections),
                    is_running=True
                )
                callback(status)

            # Optionally embed sentences in background too
            if sentence_windows and self.config.AUTO_EMBED_SENTENCES:
                logger.debug(f"[Embedding] Processing {len(sentence_windows)} sentence windows...")

                sent_embeddings = self.embed_batch([w.text for w in sentence_windows])
                self.vector_store.add_sentence_windows(
                    sentence_windows,
                    sent_embeddings,
                    document_key=document.zotero_key,
                    zotero_key=document.zotero_key,
                    parent_item_key=document.parent_item_key,
                )

            # ...existing code...

            elapsed = time.time() - start_time
            logger.debug(
                f"[Embedding] Complete: {document.title[:40]}... ({doc_key}) in {elapsed:.1f}s - "
                f"{len(sections)} sections, {len(sentence_windows) if sentence_windows else 0} sentences"
            )

        except Exception as e:
            logger.error(f"Failed to embed document {document.zotero_key}: {e}")
            if callback:
                status = EmbeddingStatus(is_running=False)
                callback(status)

    def embed_documents_async(
        self,
        documents: List[tuple[Document, str]],
        progress_callback: Optional[Callable[[str, EmbeddingStatus], None]] = None,
        continue_on_error: bool = True
    ) -> List[Future]:
        """Submit multiple documents for background embedding.
        
        Args:
            documents: List of (Document, pdf_path) tuples
            progress_callback: Optional callback for progress updates
            continue_on_error: If True, continue processing other docs even if one fails
        """
        futures = []
        for document, pdf_path in documents:
            def make_callback(doc_key: str):
                def cb(status: EmbeddingStatus):
                    if progress_callback:
                        progress_callback(doc_key, status)
                return cb

            try:
                future = self.embed_document_async(
                    document, pdf_path,
                    callback=make_callback(document.zotero_key)
                )
                futures.append(future)
            except Exception as e:
                logger.error(f"Failed to submit {document.zotero_key} for embedding: {e}")
                if not continue_on_error:
                    raise
                # Continue with next document
        
        return futures

    def embed_documents_sync(
        self,
        documents: List[tuple[Document, str]],
        stop_on_first_error: bool = False
    ) -> dict:
        """Embed multiple documents synchronously (blocking).
        
        Args:
            documents: List of (Document, pdf_path) tuples  
            stop_on_first_error: If True, raise on first error. Otherwise continue.
            
        Returns:
            Dict with success/failure counts per document key
        """
        results = {"success": [], "failed": []}
        
        for document, pdf_path in documents:
            try:
                logger.info(f"Processing: {document.title[:50]}... ({document.zotero_key})")
                
                # Process the document synchronously
                sections, sentence_windows = self.process_document(
                    document, pdf_path,
                    embed_sentences=self.config.AUTO_EMBED_SENTENCES
                )
                
                if not sections:
                    logger.warning(f"No sections extracted from {document.zotero_key}")
                    results["failed"].append(document.zotero_key)
                    continue
                    
                # Embed sections
                section_embeddings = self.embed_batch([s.text for s in sections])
                self.vector_store.add_sections(
                    sections,
                    section_embeddings,
                    document_key=document.zotero_key,
                    zotero_key=document.zotero_key,
                    parent_item_key=document.parent_item_key,
                )

                logger.info(f"Embedded {len(sections)} sections from {document.title[:40]}...")
                
                # Update metadata
                embedded_docs = self.vector_store.get_embedded_documents()
                embedded_docs[document.zotero_key] = len(sections)
                self.vector_store.save_embedded_documents(embedded_docs)

                # Optionally embed sentences
                if sentence_windows and self.config.AUTO_EMBED_SENTENCES:
                    sent_embeddings = self.embed_batch([w.text for w in sentence_windows])
                    self.vector_store.add_sentence_windows(
                        sentence_windows,
                        sent_embeddings,
                        document_key=document.zotero_key,
                        zotero_key=document.zotero_key,
                        parent_item_key=document.parent_item_key,
                    )
                    logger.info(f"Embedded {len(sentence_windows)} sentence windows")
                
                results["success"].append(document.zotero_key)
                
            except Exception as e:
                logger.error(f"Failed to embed {document.zotero_key}: {e}")
                if stop_on_first_error:
                    raise
                results["failed"].append(document.zotero_key)
                continue
                
        return results


def get_pdf_documents_from_directory(pdf_dir: Path) -> List[tuple[Document, str]]:
    """Scan a directory for PDF files and create Document objects.
    
    Args:
        pdf_dir: Directory containing PDF files
        
    Returns:
        List of (Document, path_str) tuples
    """
    documents = []
    
    if not pdf_dir.exists():
        logger.warning(f"PDF directory does not exist: {pdf_dir}")
        return documents
    
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        # Use filename (without extension) as the zotero_key
        key = pdf_path.stem
        
        doc = Document(
            zotero_key=key,
            title=pdf_path.name,  # Use filename as title fallback
            pdf_path=pdf_path
        )
        documents.append((doc, str(pdf_path)))
    
    logger.info(f"Found {len(documents)} PDFs in {pdf_dir}")
    return documents


if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Run embedding without MCP server")
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="./data/pdfs",
        help="Directory containing PDF files to embed"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true", 
        help="Stop on first error instead of continuing"
    )
    args = parser.parse_args()
    
    # Ensure directories exist
    Config.ensure_dirs()
    
    # Initialize embedding manager
    config = Config()
    manager = EmbeddingManager(config)
    
    # Get documents from PDF directory
    pdf_dir = Path(args.pdf_dir)
    documents = get_pdf_documents_from_directory(pdf_dir)
    
    if not documents:
        print(f"No PDFs found in {pdf_dir}")
        sys.exit(1)
    
    # Filter to unembedded docs
    embedded = manager.vector_store.get_embedded_documents()
    pending = [
        (doc, path) for doc, path in documents 
        if doc.zotero_key not in embedded or embedded[doc.zotero_key] == 0
    ]
    
    print(f"Total PDFs: {len(documents)}, Pending: {len(pending)}")
    
    # Embed all pending documents
    results = manager.embed_documents_sync(
        pending, 
        stop_on_first_error=args.stop_on_error
    )
    
    print(f"\nResults:")
    print(f"  Success: {len(results['success'])}")
    print(f"  Failed:  {len(results['failed'])}")
    
    if results["failed"]:
        print(f"\nFailed documents: {results['failed']}")
    
    sys.exit(0 if not results["failed"] else 1)
