"""Background embedding manager with ThreadPoolExecutor."""

import argparse
import logging
import sys
import tempfile
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

    @property
    def executor(self) -> ThreadPoolExecutor:
        """Lazy initialization of thread pool."""
        if self._executor is None:
            max_workers = getattr(self.config, 'MAX_EMBEDDING_WORKERS', 4)
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
        return self._executor

    def shutdown(self):
        """Shutdown the executor."""
        if self._executor:
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

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        # Process one at a time due to Ollama API
        embeddings = []
        for text in texts:
            emb = self.embed_text(text)
            embeddings.append(emb)
        return embeddings

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _rerank(self, query: str, candidates: List[str]) -> List[float]:
        """Rerank candidates using Ollama reranker."""
        response = ollama.generate(
            model=self.config.RERANKER_MODEL,
            prompt=f"Given the query '{query}', rate each document from 0-1 relevance:\n\n" +
                   "\n".join([f"{i}: {doc[:200]}..." for i, doc in enumerate(candidates)]),
            options={"num_ctx": 32768}
        )
        # Parse scores from response - simplified implementation
        return [0.9] * len(candidates)  # Placeholder: actual impl needs score parsing

    # --- Document processing ---

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
            logger.error(f"Failed to fetch PDF for {doc_key} from Zotero")
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
            
            logger.info(f"[Embedding] Starting: {document.title[:50]}... ({doc_key})")

            # Extract sections (always do this)
            sections, sentence_windows = self.process_document(
                document, str(temp_path),
                embed_sentences=self.config.AUTO_EMBED_SENTENCES
            )
            
            logger.info(f"[Embedding] Extracted {len(sections)} sections from {document.title[:30]}...")

            # Embed sections
            section_embeddings = self.embed_batch([s.text for s in sections])
            self.vector_store.add_sections(sections, section_embeddings, document.zotero_key)
            
            logger.info(f"[Embedding] Embedded {len(sections)} sections")

            # Update metadata
            embedded_docs = self.vector_store.get_embedded_documents()
            embedded_docs[document.zotero_key] = len(sections)
            self.vector_store.save_embedded_documents(embedded_docs)

            if callback:
                status = EmbeddingStatus(
                    embedded_sections=len(sections),
                    is_running=True
                )
                callback(status)

            # Optionally embed sentences in background too
            if sentence_windows and self.config.AUTO_EMBED_SENTENCES:
                logger.info(f"[Embedding] Processing {len(sentence_windows)} sentence windows...")
                
                sent_embeddings = self.embed_batch([w.text for w in sentence_windows])
                self.vector_store.add_sentence_windows(
                    sentence_windows, sent_embeddings, document.zotero_key
                )
                
                if callback:
                    status = EmbeddingStatus(
                        embedded_sections=len(sections),
                        embedded_sentences=len(sentence_windows),
                        is_running=False
                    )
                    callback(status)
            elif not sentence_windows:
                if callback:
                    status = EmbeddingStatus(
                        embedded_sections=len(sections),
                        is_running=False
                    )
                    callback(status)

            elapsed = time.time() - start_time
            logger.info(f"[Embedding] Complete: {document.title[:40]}... ({doc_key}) in {elapsed:.1f}s - {len(sections)} sections, {len(sentence_windows) if sentence_windows else 0} sentences")

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

            logger.info(f"[Embedding] Starting: {document.title[:50]}... ({doc_key})")

            # Extract sections (always do this)
            sections, sentence_windows = self.process_document(
                document, pdf_path,
                embed_sentences=self.config.AUTO_EMBED_SENTENCES
            )
            
            logger.info(f"[Embedding] Extracted {len(sections)} sections from {document.title[:30]}...")

            # Embed sections
            section_embeddings = self.embed_batch([s.text for s in sections])
            self.vector_store.add_sections(sections, section_embeddings, document.zotero_key)
            
            logger.info(f"[Embedding] Embedded {len(sections)} sections")

            # Update metadata
            embedded_docs = self.vector_store.get_embedded_documents()
            embedded_docs[document.zotero_key] = len(sections)
            self.vector_store.save_embedded_documents(embedded_docs)

            if callback:
                status = EmbeddingStatus(
                    embedded_sections=len(sections),
                    is_running=True
                )
                callback(status)

            # Optionally embed sentences in background too
            if sentence_windows and self.config.AUTO_EMBED_SENTENCES:
                logger.info(f"[Embedding] Processing {len(sentence_windows)} sentence windows...")
                
                sent_embeddings = self.embed_batch([w.text for w in sentence_windows])
                self.vector_store.add_sentence_windows(
                    sentence_windows, sent_embeddings, document.zotero_key
                )
                
                if callback:
                    status = EmbeddingStatus(
                        embedded_sections=len(sections),
                        embedded_sentences=len(sentence_windows),
                        is_running=False
                    )
                    callback(status)
            elif not sentence_windows:
                if callback:
                    status = EmbeddingStatus(
                        embedded_sections=len(sections),
                        is_running=False
                    )
                    callback(status)

            elapsed = time.time() - start_time
            logger.info(f"[Embedding] Complete: {document.title[:40]}... ({doc_key}) in {elapsed:.1f}s - {len(sections)} sections, {len(sentence_windows) if sentence_windows else 0} sentences")

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
                self.vector_store.add_sections(sections, section_embeddings, document.zotero_key)
                
                logger.info(f"Embedded {len(sections)} sections from {document.title[:40]}...")
                
                # Update metadata
                embedded_docs = self.vector_store.get_embedded_documents()
                embedded_docs[document.zotero_key] = len(sections)
                self.vector_store.save_embedded_documents(embedded_docs)

                # Optionally embed sentences
                if sentence_windows and self.config.AUTO_EMBED_SENTENCES:
                    sent_embeddings = self.embed_batch([w.text for w in sentence_windows])
                    self.vector_store.add_sentence_windows(
                        sentence_windows, sent_embeddings, document.zotero_key
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
