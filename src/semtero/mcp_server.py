"""MCP server for SemTero."""

import logging
import threading
from typing import Optional, Callable, Any

from fastmcp import FastMCP

from semtero.config import Config
from semtero.doi_client import DoiClient, normalize_doi
from semtero.embedding_manager import EmbeddingManager
from semtero.search_engine import SearchEngine
from semtero.zotero_client import ZoteroClient
from semtero.models import CitationReturnMode, SearchResult
from semtero.reranker import Reranker
from semtero.vector_store import VectorStore

logger = logging.getLogger(__name__)


class MCPZoteroServer:
    """MCP server exposing SemTero tools."""

    def __init__(self, config: Config):
        self.config = config
        self.zotero_client = ZoteroClient(api_url=config.ZOTERO_API_URL)
        self.doi_client = DoiClient(
            base_url=config.DOI_BIBTEX_BASE_URL,
            timeout_seconds=float(config.DOI_BIBTEX_TIMEOUT_SECONDS),
        )
        shared_vector_store = VectorStore(str(config.VECTOR_STORE_DIR))
        self.embedding_manager = EmbeddingManager(
            config,
            vector_store=shared_vector_store,
        )
        self.search_engine = SearchEngine(
            config,
            vector_store=shared_vector_store,
        )
        # Cache for document metadata to avoid repeated API calls
        self._metadata_cache: dict[str, dict] = {}
        self._auto_embed_done = False
        self._embedding_lock = threading.Lock()
        self._embedding_thread: threading.Thread | None = None
        self._status_listeners: list[Callable[[object], None]] = []
        self._last_run_summary: str = "Idle"
        self._next_auto_reembed_at: str = ""

    def register_embedding_status_listener(
        self, listener: Callable[[object], None]
    ) -> None:
        """Register a callback notified whenever embedding progress changes."""

        self._status_listeners.append(listener)

    def set_next_auto_reembed_at(self, timestamp: str) -> None:
        """Expose the next scheduled auto re-embed time for the UI/API."""

        self._next_auto_reembed_at = timestamp

    def _notify_embedding_status(self, status=None) -> None:
        status = status or self.embedding_manager.get_embedding_status()
        for listener in list(self._status_listeners):
            try:
                listener(status)
            except Exception:
                logger.debug("Embedding status listener failed", exc_info=True)

    def _is_document_pending(self, doc, embedded: dict[str, int]) -> bool:
        if doc.zotero_key in embedded:
            return False

        try:
            return not self.embedding_manager.vector_store.is_document_embedded(
                doc.zotero_key
            )
        except Exception:
            logger.debug(
                "Falling back to metadata-only embedded check for %s",
                doc.zotero_key,
                exc_info=True,
            )
            return True

    def _get_pending_documents(self) -> tuple[list, int]:
        embedded = self.embedding_manager.vector_store.get_embedded_documents()
        pending = []
        total_available = 0
        for doc in self.zotero_client.get_documents_with_pdfs():
            total_available += 1
            if self._is_document_pending(doc, embedded):
                pending.append(doc)

        return pending, total_available

    def _finish_embedding_job(self, *, last_error: str = ""):
        finish_job = getattr(self.embedding_manager, "finish_embedding_job", None)
        if finish_job is None:
            raise AttributeError("EmbeddingManager is missing finish_embedding_job")
        if last_error:
            return finish_job(last_error=last_error)
        return finish_job()

    def _run_background_embedding(self, trigger: str) -> None:
        try:
            scan_status = self.embedding_manager.mark_embedding_scan_started()
            self._next_auto_reembed_at = ""
            self._last_run_summary = (
                f"Scanning Zotero for documents to embed (trigger: {trigger})."
            )
            self._notify_embedding_status(scan_status)

            embedded = self.embedding_manager.vector_store.get_embedded_documents()
            futures = []
            total_available = 0
            pending_count = 0

            for doc in self.zotero_client.get_documents_with_pdfs():
                total_available += 1
                scan_status = self.embedding_manager.set_embedding_job_total(
                    total_available
                )
                is_pending = self._is_document_pending(doc, embedded)

                if not is_pending:
                    scan_status = self.embedding_manager.mark_document_completed()
                    if total_available == 1 or total_available % 25 == 0:
                        self._last_run_summary = (
                            "Scanning Zotero for PDFs. "
                            f"Seen {total_available} document(s), {pending_count} pending embedding so far."
                        )
                        self._notify_embedding_status(scan_status)
                    continue

                pending_count += 1
                if total_available == 1 or total_available % 25 == 0:
                    self._last_run_summary = (
                        "Scanning Zotero for PDFs. "
                        f"Seen {total_available} document(s), {pending_count} pending embedding so far."
                    )
                    self._notify_embedding_status(scan_status)

                try:
                    futures.append(
                        self.embedding_manager.embed_document_async_with_client(
                            doc,
                            self.zotero_client,
                            callback=self._notify_embedding_status,
                        )
                    )
                except Exception as e:
                    logger.error(
                        "Failed to submit %s for embedding: %s", doc.zotero_key, e
                    )
                    snapshot = self.embedding_manager.mark_document_completed(
                        failed=True,
                        last_error=f"Failed to submit {doc.zotero_key}: {e}",
                    )
                    self._notify_embedding_status(snapshot)

            if total_available == 0:
                self._last_run_summary = "No Zotero documents with PDFs were found."
                final_status = self._finish_embedding_job()
                self._notify_embedding_status(final_status)
                return

            if not pending_count:
                self._last_run_summary = (
                    "All PDF documents were already embedded or previously processed."
                )
                final_status = self._finish_embedding_job()
                self._notify_embedding_status(final_status)
                return

            self._last_run_summary = (
                f"Embedding {pending_count} new document(s) triggered by {trigger}."
            )
            self._notify_embedding_status(
                self.embedding_manager.set_embedding_job_total(total_available)
            )

            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.warning("Background embedding future failed: %s", e)

            final_status = self._finish_embedding_job()
            if final_status.failed_documents:
                self._last_run_summary = (
                    f"Embedding finished with {final_status.failed_documents} failed document(s)."
                )
            else:
                self._last_run_summary = (
                    "Embedding finished successfully for "
                    f"{final_status.processed_documents}/{final_status.total_documents} document(s)."
                )
            self._notify_embedding_status(final_status)
        except Exception as e:
            logger.exception("Background embedding run failed")
            final_status = self._finish_embedding_job(last_error=str(e))
            self._last_run_summary = f"Embedding failed: {e}"
            self._notify_embedding_status(final_status)
        finally:
            with self._embedding_lock:
                self._embedding_thread = None

    def start_background_embedding(self, trigger: str = "manual") -> dict:
        """Start embedding any new documents in a background thread."""

        with self._embedding_lock:
            current_status = self.embedding_manager.get_embedding_status()
            if current_status.is_running or (
                self._embedding_thread and self._embedding_thread.is_alive()
            ):
                return {
                    "status": "already_running",
                    "message": "A background embedding job is already running.",
                }

            self._embedding_thread = threading.Thread(
                target=self._run_background_embedding,
                args=(trigger,),
                daemon=True,
            )
            self._embedding_thread.start()

        return {
            "status": "started",
            "message": "Background embedding scan started.",
            "trigger": trigger,
        }

    async def start_auto_embedding(self):
        """Auto-start background embedding on server startup."""
        if self._auto_embed_done:
            return

        try:
            result = self.start_background_embedding(trigger="startup")
            if result.get("status") == "already_running":
                logger.info(result.get("message"))
        except Exception as e:
            logger.warning(f"Auto-embedding failed (non-blocking): {e}")

        self._auto_embed_done = True

    def _get_metadata_for_key(self, item_key: str) -> dict:
        """Get cached or fresh metadata for an item."""
        logger.debug(f"_get_metadata_for_key called with key: {item_key}")

        if item_key in self._metadata_cache:
            logger.debug(f"Found metadata in cache for key: {item_key}")
            return self._metadata_cache[item_key]

        # Try to get from Zotero
        logger.debug(f"Calling zotero_client.get_item_metadata({item_key})")
        metadata = self.zotero_client.get_item_metadata(item_key)
        logger.debug(f"Got metadata result: {metadata}")

        if metadata:
            self._metadata_cache[item_key] = metadata
            return metadata

        # Return empty metadata structure
        logger.debug(
            f"No metadata found, returning empty structure for key: {item_key}"
        )
        return {
            "bibtex": "",
            "file_path": "",
            "title": "",
            "authors": [],
            "date": "",
            "item_type": "",
        }

    # --- MCP Tool Implementations ---

    def do_reranking(
        self, results: list[SearchResult], query: str
    ) -> list[SearchResult]:
        """Optional second-stage reranking of search results."""

        rer = Reranker(
            model_name=self.config.RERANKER_MODEL,
            min_gpu_vram_gb=self.config.RERANKER_GPU_MIN_VRAM_GB,
        )
        try:
            return rer.rerank(results, query)
        finally:
            rer.release_device()

    async def search_documents(
        self,
        query: str,
        document_key: Optional[str] = None,
        top_sentences: int = 10,
        min_relevance: float = 0.75,
        citation_return_mode: CitationReturnMode = "sentence",
        require_cited_bibtex: bool = False,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> list[dict]:
        """Search across embedded documents.

        Args:
            citation_return_mode:
                - "sentence": return sentence only
                - "bibtex": return only cited BibTeX entries for the matched sentence
                - "both": return sentence plus cited BibTeX entries
            require_cited_bibtex:
                If True, only return results where the matched sentence has at least
                one cited BibTeX entry attached.

        Returns enriched results including Zotero item BibTeX/file metadata as well as
        per-sentence citation metadata.
        """
        logger.debug(f"search_documents called with query: {query}")

        def emit_progress(payload: dict[str, Any]) -> None:
            if not progress_callback:
                return
            try:
                percentage = payload.get("percentage")
                if percentage is not None:
                    payload = {
                        **payload,
                        "percentage": max(0.0, min(100.0, float(percentage))),
                    }
                progress_callback(payload)
            except Exception:
                logger.debug("Search progress callback failed", exc_info=True)

        temp_top_sentences = max(500, top_sentences)

        emit_progress(
            {
                "stage": "start",
                "percentage": 0,
                "message": "Starting search",
                "detail": "Initializing search across embedded documents",
                "result_count": 0,
            }
        )

        results = self.search_engine.search_best_sentences(
            query=query,
            document_key=document_key,
            top_sentences=temp_top_sentences,
            citation_return_mode=citation_return_mode,
            progress_callback=emit_progress,
        )

        logger.debug(f"Initial search returned {len(results)} results before filtering")

        if require_cited_bibtex:
            before = len(results)
            results = [r for r in results if getattr(r, "cited_bibtex", None)]
            logger.debug(f"Filtered require_cited_bibtex: {before} -> {len(results)}")

        ret = []
        # Remove results below threshold:
        for r in results:
            if r.relevance_score >= min_relevance:
                logger.debug(
                    f"Keeping result with relevance {r.relevance_score}, sentence: {r.text[:150]}..."
                )
                ret.append(r)
            else:
                logger.debug(
                    f"Filtering out result with relevance {r.relevance_score} below threshold {min_relevance}, sentence: {r.text[:150]}..."
                )

        if len(ret) == 0:
            logger.debug("No results above relevance threshold")
            emit_progress(
                {
                    "stage": "complete",
                    "percentage": 100,
                    "message": "No results found",
                    "detail": "No sentences remained after the relevance filters",
                    "result_count": 0,
                }
            )
            return []

        emit_progress(
            {
                "stage": "reranking",
                "percentage": 60,
                "message": "Performing reranking",
                "detail": f"Reranking {len(ret)} candidate sentence{'s' if len(ret) != 1 else ''}",
                "candidate_sentences": len(ret),
            }
        )

        # Do reranking
        results = self.do_reranking(ret, query)

        logger.debug(f"Got {len(results)} search results")

        #keeping only top_sentences after reranking and filtering
        results = results[:top_sentences]
        logger.debug(f"Trimmed to top_sentences={top_sentences}: {len(results)} results")

        # First pass: collect all unique keys and fetch metadata for each once
        # This ensures we call Zotero API at most once per unique document
        key_to_metadata: dict[str, dict] = {}
        metadata_keys = [
            r.zotero_key for r in results if r.zotero_key and r.zotero_key not in key_to_metadata
        ]
        total_metadata = len(metadata_keys)

        emit_progress(
            {
                "stage": "metadata",
                "percentage": 78 if total_metadata else 95,
                "message": f"Gathering Final Metadata 0 of {total_metadata}",
                "detail": "Fetching Zotero metadata for the final result documents",
                "metadata_current": 0,
                "metadata_total": total_metadata,
            }
        )

        for index, zotero_key in enumerate(metadata_keys, start=1):
            logger.debug(f"Processing result with zotero_key: {zotero_key}")

            meta = self._get_metadata_for_key(zotero_key)

            # If no Zotero title, fall back to vector store title
            if not meta.get("title"):
                vs_title = self.search_engine.vector_store.get_document_title(
                    zotero_key
                )
                if vs_title:
                    meta["title"] = vs_title

            key_to_metadata[zotero_key] = meta

            emit_progress(
                {
                    "stage": "metadata",
                    "percentage": 78 + ((17 * index) / total_metadata) if total_metadata else 95,
                    "message": f"Gathering Final Metadata {index} of {total_metadata}",
                    "detail": f"Loaded metadata for document {index} of {total_metadata}",
                    "metadata_current": index,
                    "metadata_total": total_metadata,
                }
            )

        # Second pass: apply metadata to ALL results (not just the first one per key)
        enriched_results = []

        for r in results:
            result_dict = r.to_dict()

            zotero_key = r.zotero_key

            if zotero_key and zotero_key in key_to_metadata:
                meta = key_to_metadata[zotero_key]

                # Apply all enriched data to this result
                result_dict["bibtex"] = meta.get("bibtex", "")
                result_dict["file_path"] = meta.get("file_path", "")
                result_dict["authors"] = meta.get("authors", [])
                result_dict["date"] = meta.get("date", "")
                result_dict["item_type"] = meta.get("item_type", "")

                # Set document_title - prefer Zotero title, fallback to vector store
                doc_title = meta.get("title", "")
                if not doc_title:
                    vs_title = self.search_engine.vector_store.get_document_title(
                        zotero_key
                    )
                    if vs_title:
                        doc_title = vs_title
                result_dict["document_title"] = doc_title

            # If we still don't have a document title, use section info as fallback
            if not result_dict.get("document_title") and result_dict.get(
                "section_title"
            ):
                result_dict["document_title"] = (
                    f"Document containing: {result_dict['section_title'][:50]}"
                )

            enriched_results.append(result_dict)

        emit_progress(
            {
                "stage": "complete",
                "percentage": 100,
                "message": f"Found {len(enriched_results)} result{'s' if len(enriched_results) != 1 else ''}",
                "detail": "Search complete",
                "result_count": len(enriched_results),
            }
        )

        return enriched_results

    async def get_library_items(self, limit: int = 25) -> list[dict]:
        """Get items from Zotero library."""
        items = self.zotero_client.get_items(limit=limit)
        return [
            {
                "key": item.get("key", ""),
                "title": item.get("data", {}).get("title", ""),
                "authors": [
                    a.get("firstName", "") + " " + a.get("lastName", "")
                    for a in item.get("data", {}).get("creators", [])
                ],
                "date": item.get("data", {}).get("date", ""),
            }
            for item in items
        ]

    async def get_documents_with_pdfs(self) -> list[dict]:
        """Get all documents that have PDFs."""
        docs = self.zotero_client.get_documents_with_pdfs()
        return [
            {
                "key": doc.zotero_key,
                "title": doc.title,
                "authors": doc.authors,
                "has_pdf": bool(doc.pdf_path),
            }
            for doc in docs
        ]

    async def embed_new_documents_now(self) -> dict:
        """Trigger embedding of newly discovered documents immediately."""

        return self.start_background_embedding(trigger="manual")

    async def start(self):
        """Start the MCP server with auto-embedding."""
        # Trigger background embedding (non-blocking)
        await self.start_auto_embedding()

    async def sync_and_embed(
        self, embed_sentences: bool = False, document_key: Optional[str] = None
    ) -> dict:
        """Sync from Zotero and optionally embed new documents.

        Uses on-demand PDF fetching - PDFs are fetched temporarily for embedding
        and not saved to disk permanently.
        """
        # Get documents with PDFs
        all_docs_with_pdfs = list(self.zotero_client.get_documents_with_pdfs())

        if not all_docs_with_pdfs:
            return {
                "status": "no_documents",
                "message": "No documents with PDFs found in Zotero",
            }

        # Filter by document key if specified
        docs_to_process = [
            doc
            for doc in all_docs_with_pdfs
            if document_key is None or doc.zotero_key == document_key
        ]

        if not docs_to_process:
            return {
                "status": "no_match",
                "message": f"No documents matching {document_key}",
            }

        # Get currently embedded docs
        embedded = self.embedding_manager.vector_store.get_embedded_documents()

        # Filter to only new/updated documents (using zotero_client approach)
        pending = [
            doc for doc in docs_to_process if self._is_document_pending(doc, embedded)
        ]

        if not pending:
            return {
                "status": "up_to_date",
                "message": f"All {len(docs_to_process)} documents already embedded",
            }

        # Submit for background embedding using on-demand PDF fetching
        # This uses the new method that fetches PDFs temporarily without saving
        futures = []
        for doc in pending:

            def make_callback(doc_key: str):
                def cb(status):
                    self._progress_callback(doc_key, status)

                return cb

            try:
                future = self.embedding_manager.embed_document_async_with_client(
                    doc, self.zotero_client, callback=make_callback(doc.zotero_key)
                )
                futures.append(future)
            except Exception as e:
                logger.error(f"Failed to submit {doc.zotero_key} for embedding: {e}")

        return {
            "status": "started",
            "pending_count": len(pending),
            "message": f"Started embedding {len(pending)} documents (temp file mode)",
        }

    def _progress_callback(self, doc_key: str, status):
        """Handle embedding progress updates."""
        logger.info(f"Document {doc_key}: sections={status.embedded_sections}")

    async def get_embedding_status(self) -> dict:
        """Get current embedding statistics."""
        stats = self.search_engine.get_stats()
        embedded_docs = self.embedding_manager.vector_store.get_embedded_documents()
        progress = self.embedding_manager.get_embedding_status()
        return {
            **stats,
            **progress.to_dict(),
            "documents": sorted(embedded_docs.keys()),
            "state": "running" if progress.is_running else "idle",
            "last_run_summary": self._last_run_summary,
            "auto_reembed_interval_minutes": self.config.AUTO_REEMBED_INTERVAL_MINUTES,
            "next_auto_reembed_at": self._next_auto_reembed_at,
        }

    async def delete_document(self, document_key: str) -> dict:
        """Delete all embeddings for a document."""
        self.embedding_manager.vector_store.delete_document(document_key)

        # Update metadata
        embedded = self.embedding_manager.vector_store.get_embedded_documents()
        if document_key in embedded:
            del embedded[document_key]
            self.embedding_manager.vector_store.save_embedded_documents(embedded)

        return {"status": "deleted", "document_key": document_key}

    async def reembed_document(self, document_key: str) -> dict:
        """Re-embed a specific document using on-demand PDF fetching."""
        docs = list(self.zotero_client.get_documents_with_pdfs())
        target_doc = None
        for doc in docs:
            if doc.zotero_key == document_key:
                target_doc = doc
                break

        if not target_doc:
            return {"status": "error", "message": f"Document {document_key} not found"}

        # Delete existing embeddings first
        self.embedding_manager.vector_store.delete_document(document_key)

        embedded = self.embedding_manager.vector_store.get_embedded_documents()
        if document_key in embedded:
            del embedded[document_key]
            self.embedding_manager.vector_store.save_embedded_documents(embedded)

        # Re-embed using on-demand PDF fetching (temp file mode)
        self.embedding_manager.embed_document_async_with_client(
            target_doc, self.zotero_client
        )

        return {"status": "reembedding", "document_key": document_key}

    async def import_item_by_doi(
        self, doi: str, collection_key: Optional[str] = None
    ) -> dict:
        """Import a bibliographic item into Zotero by DOI.

        Accepts a bare DOI ("10.1111/cgf.13217") or a DOI URL ("https://doi.org/... ").
        """
        try:
            norm = normalize_doi(doi)
        except ValueError as e:
            return {"status": "error", "message": str(e)}

        try:
            bibtex = self.doi_client.fetch_bibtex(norm)
        except Exception as e:
            return {
                "status": "error",
                "doi": norm,
                "message": f"Failed to fetch BibTeX: {e}",
            }

        result = self.zotero_client.import_bibtex_via_connector(
            bibtex,
            collection_key=collection_key,
            timeout_seconds=self.config.ZOTERO_CONNECTOR_TIMEOUT_SECONDS,
        )
        result.setdefault("doi", norm)
        # Helpful for debugging but not too noisy: only include bibtex length.
        result.setdefault("bibtex_chars", len(bibtex))
        return result

    def shutdown(self):
        """Shutdown the server."""
        embedding_thread = self._embedding_thread
        if (
            embedding_thread
            and embedding_thread.is_alive()
            and threading.current_thread() is not embedding_thread
        ):
            embedding_thread.join(timeout=5)

        self.embedding_manager.shutdown()

        try:
            self.zotero_client.close()
        except Exception:
            logger.debug("Failed to close Zotero client session", exc_info=True)

        try:
            self.doi_client.close()
        except Exception:
            logger.debug("Failed to close DOI client session", exc_info=True)


# --- FastMCP wiring ---

mcp = FastMCP("SemTero")

# Global server instance, set by main.py during startup.
_SERVER: MCPZoteroServer | None = None


def set_server_instance(server: MCPZoteroServer) -> None:
    """Register the live server instance used by FastMCP tool wrappers."""
    global _SERVER
    _SERVER = server


def get_server() -> MCPZoteroServer:
    """Get the registered server instance or raise a helpful error."""
    if _SERVER is None:
        raise RuntimeError(
            "MCPZoteroServer instance not registered. "
            "Call set_server_instance(MCPZoteroServer(...)) before starting FastMCP."
        )
    return _SERVER


@mcp.tool
async def search_documents(
    query: str,
    document_key: Optional[str] = None,
    top_sentences: int = 10,
    min_relevance: float = 0.75,
    citation_return_mode: CitationReturnMode = "sentence",
    require_cited_bibtex: bool = False,
) -> list[dict]:
    """
    Search Zotero for relevant sentences matching the query, with enriched metadata and optional citation info.
    :param query: The search query string.
    :param document_key: Optional - Zotero document key to restrict the search to a specific document.
    :param top_sentences: The maximum number of sentences to return.
    :param min_relevance: Minimum relevance score (0 to 1) for returned sentences. Keep this to at least 0.75 to ensure quality.
    :param citation_return_mode: Determines how to return citation info for matched sentences:
        - "sentence": return the matched sentence text only (default)
        - "bibtex": return only the BibTeX entries for citations attached to the matched sentence
        - "both": return both the matched sentence and its attached BibTeX entries
    :param require_cited_bibtex: If True, only return sentences that have at least one cited BibTeX entry attached. This can be used to filter for higher-quality results that have explicit citations.
    :return: A list of dictionaries, each containing:
        - text: The matched sentence text
        - document_title: The title of the document containing the sentence
        - section_title: The title of the section containing the sentence (if available)
        - zotero_key: The Zotero key of the document containing the sentence
        - relevance_score: The relevance score of the sentence to the query (0 to 1)
        - cited_bibtex: A list of BibTeX entries for citations attached to the sentence (if citation_return_mode is "bibtex" or "both")
        - bibtex: The BibTeX entry for the document (if available)
        - file_path: The file path to the document's PDF (if available)
        - authors: A list of authors for the document (if available)
        - date: The publication date of the document (if available)
        - item_type: The Zotero item type (e.g. "book", "article") (if available)
    """
    logger.debug(f"MCP tool search_documents called with query: {query}")
    """Search across embedded documents using two-stage RAG."""
    return await get_server().search_documents(
        query=query,
        document_key=document_key,
        top_sentences=top_sentences,
        min_relevance=min_relevance,
        citation_return_mode=citation_return_mode,
        require_cited_bibtex=require_cited_bibtex,
    )


#
# @mcp.tool
# async def get_library_items(limit: int = 25) -> list[dict]:
#     """Get items from Zotero library."""
#     return await get_server().get_library_items(limit=limit)
#
#
# @mcp.tool
# async def get_documents_with_pdfs() -> list[dict]:
#     """Get all documents that have PDFs."""
#     return await get_server().get_documents_with_pdfs()
#
#
# @mcp.tool
# async def sync_and_embed(embed_sentences: bool = False, document_key: Optional[str] = None) -> dict:
#     """Sync from Zotero and optionally embed new documents."""
#     return await get_server().sync_and_embed(embed_sentences=embed_sentences, document_key=document_key)
#
#
# @mcp.tool
# async def get_embedding_status() -> dict:
#     """Get current embedding statistics."""
#     return await get_server().get_embedding_status()
#
#
# @mcp.tool
# async def delete_document(document_key: str) -> dict:
#     """Delete all embeddings for a document."""
#     return await get_server().delete_document(document_key)
#
#
# @mcp.tool
# async def reembed_document(document_key: str) -> dict:
#     """Re-embed a specific document."""
#     return await get_server().reembed_document(document_key)


@mcp.tool
async def import_item_by_doi(doi: str, collection_key: Optional[str] = None) -> dict:
    """Fetch BibTeX for a DOI and import the item into Zotero via the Connector."""
    return await get_server().import_item_by_doi(doi=doi, collection_key=collection_key)


def main() -> None:
    """Run the FastMCP server (stdio)."""
    from semtero.logging_setup import setup_logging

    setup_logging()

    config = Config()
    server = MCPZoteroServer(config)
    set_server_instance(server)

    logger.info("FastMCP server starting with stdio transport...")

    # Note: background embedding is started by main.py in the full app.
    # Keeping this entrypoint minimal for standalone use.
    mcp.run()


if __name__ == "__main__":
    main()
