"""MCP server for ZoteroRAG."""

import logging
from typing import Optional

from fastmcp import FastMCP

from zoterorag.config import Config
from zoterorag.doi_client import DoiClient, normalize_doi
from zoterorag.embedding_manager import EmbeddingManager
from zoterorag.search_engine import SearchEngine
from zoterorag.zotero_client import ZoteroClient
from zoterorag.models import CitationReturnMode, SearchResult
from zoterorag.reranker import Reranker

logger = logging.getLogger(__name__)
#set level to debug
logger.setLevel(logging.DEBUG)


class MCPZoteroServer:
    """MCP server exposing Zotero RAG tools."""

    def __init__(self, config: Config):
        self.config = config
        self.zotero_client = ZoteroClient(api_url=config.ZOTERO_API_URL)
        self.doi_client = DoiClient(
            base_url=config.DOI_BIBTEX_BASE_URL,
            timeout_seconds=float(config.DOI_BIBTEX_TIMEOUT_SECONDS),
        )
        self.embedding_manager = EmbeddingManager(config)
        self.search_engine = SearchEngine(config)
        # Cache for document metadata to avoid repeated API calls
        self._metadata_cache: dict[str, dict] = {}
        self._auto_embed_done = False

    async def start_auto_embedding(self):
        """Auto-start background embedding on server startup."""
        if self._auto_embed_done:
            return
        
        try:
            import asyncio
            result = await self.sync_and_embed()
            
            # Log the status but don't block startup
            if result.get("status") == "started":
                logger.info(f"Auto-embedding started: {result.get('pending_count')} documents pending")
            elif result.get("status") == "up_to_date":
                logger.info("All documents already embedded")
            else:
                logger.info(f"Sync status: {result.get('status')}")
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
        logger.debug(f"No metadata found, returning empty structure for key: {item_key}")
        return {
            "bibtex": "",
            "file_path": "",
            "title": "",
            "authors": [],
            "date": "",
            "item_type": ""
        }

    # --- MCP Tool Implementations ---

    def do_reranking(self, results: list[SearchResult], query: str) -> list[SearchResult]:
        """Optional second-stage reranking of search results."""

        rer = Reranker()
        results = rer.rerank(results,query)
        return results

    async def search_documents(
        self,
        query: str,
        document_key: Optional[str] = None,
        top_sentences: int = 10,
        min_relevance: float = 0.75,
        citation_return_mode: CitationReturnMode = "sentence",
        require_cited_bibtex: bool = False,
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

        temp_top_sentences = top_sentences * 10 if require_cited_bibtex else top_sentences

        results = self.search_engine.search_best_sentences(
            query=query,
            document_key=document_key,
            top_sentences=temp_top_sentences,
            citation_return_mode=citation_return_mode,
        )

        if require_cited_bibtex:
            before = len(results)
            results = [r for r in results if getattr(r, "cited_bibtex", None)]
            results = results[:top_sentences]
            logger.debug(f"Filtered require_cited_bibtex: {before} -> {len(results)}")

        ret = []
        # Remove results below threshold:
        for r in results:
            if r.relevance_score >= min_relevance:
                logger.debug(
                    f"Keeping result with relevance {r.relevance_score}, sentence: {r.text[:150]}...")
                ret.append(r)
            else:
                logger.debug(
                    f"Filtering out result with relevance {r.relevance_score} below threshold {min_relevance}, sentence: {r.text[:150]}...")

        # Do reranking
        results = self.do_reranking(ret, query)



        logger.debug(f"Got {len(results)} search results")
        
        # First pass: collect all unique keys and fetch metadata for each once
        # This ensures we call Zotero API at most once per unique document
        key_to_metadata: dict[str, dict] = {}
        
        for r in results:
            zotero_key = r.zotero_key
            logger.debug(f"Processing result with zotero_key: {zotero_key}")
            
            if zotero_key and zotero_key not in key_to_metadata:
                # Fetch Zotero metadata (only once per unique document)
                meta = self._get_metadata_for_key(zotero_key)
                
                # If no Zotero title, fall back to vector store title
                if not meta.get("title"):
                    vs_title = self.search_engine.vector_store.get_document_title(zotero_key)
                    if vs_title:
                        meta["title"] = vs_title
                
                key_to_metadata[zotero_key] = meta
        
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
                    vs_title = self.search_engine.vector_store.get_document_title(zotero_key)
                    if vs_title:
                        doc_title = vs_title
                result_dict["document_title"] = doc_title
            
            # If we still don't have a document title, use section info as fallback
            if not result_dict.get("document_title") and result_dict.get("section_title"):
                result_dict["document_title"] = f"Document containing: {result_dict['section_title'][:50]}"
            
            enriched_results.append(result_dict)


        return enriched_results

    async def get_library_items(self, limit: int = 25) -> list[dict]:
        """Get items from Zotero library."""
        items = self.zotero_client.get_items(limit=limit)
        return [
            {
                "key": item.get("key", ""),
                "title": item.get("data", {}).get("title", ""),
                "authors": [a.get("firstName", "") + " " + a.get("lastName", "")
                           for a in item.get("data", {}).get("creators", [])],
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

    async def start(self):
        """Start the MCP server with auto-embedding."""
        # Trigger background embedding (non-blocking)
        await self.start_auto_embedding()

    async def sync_and_embed(
        self,
        embed_sentences: bool = False,
        document_key: Optional[str] = None
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
                "message": "No documents with PDFs found in Zotero"
            }

        # Filter by document key if specified
        docs_to_process = [
            doc for doc in all_docs_with_pdfs
            if document_key is None or doc.zotero_key == document_key
        ]

        if not docs_to_process:
            return {
                "status": "no_match",
                "message": f"No documents matching {document_key}"
            }

        # Get currently embedded docs
        embedded = self.embedding_manager.vector_store.get_embedded_documents()

        # Filter to only new/updated documents (using zotero_client approach)
        pending = [
            doc for doc in docs_to_process
            if doc.zotero_key not in embedded or embedded[doc.zotero_key] == 0
        ]

        if not pending:
            return {
                "status": "up_to_date",
                "message": f"All {len(docs_to_process)} documents already embedded"
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
                    doc,
                    self.zotero_client,
                    callback=make_callback(doc.zotero_key)
                )
                futures.append(future)
            except Exception as e:
                logger.error(f"Failed to submit {doc.zotero_key} for embedding: {e}")

        return {
            "status": "started",
            "pending_count": len(pending),
            "message": f"Started embedding {len(pending)} documents (temp file mode)"
        }

    def _progress_callback(self, doc_key: str, status):
        """Handle embedding progress updates."""
        logger.info(f"Document {doc_key}: sections={status.embedded_sections}")

    async def get_embedding_status(self) -> dict:
        """Get current embedding statistics."""
        stats = self.search_engine.get_stats()
        embedded_docs = self.embedding_manager.vector_store.get_embedded_documents()
        return {
            **stats,
            "documents": list(embedded_docs.keys())
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
            target_doc,
            self.zotero_client
        )

        return {"status": "reembedding", "document_key": document_key}

    async def import_item_by_doi(self, doi: str, collection_key: Optional[str] = None) -> dict:
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
            return {"status": "error", "doi": norm, "message": f"Failed to fetch BibTeX: {e}"}

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
        self.embedding_manager.shutdown()


# --- FastMCP wiring ---

mcp = FastMCP("ZoteroRAG")

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
    min_relevance: float = 0.7,
    citation_return_mode: CitationReturnMode = "sentence",
    require_cited_bibtex: bool = False,
) -> list[dict]:
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


@mcp.tool
async def get_library_items(limit: int = 25) -> list[dict]:
    """Get items from Zotero library."""
    return await get_server().get_library_items(limit=limit)


@mcp.tool
async def get_documents_with_pdfs() -> list[dict]:
    """Get all documents that have PDFs."""
    return await get_server().get_documents_with_pdfs()


@mcp.tool
async def sync_and_embed(embed_sentences: bool = False, document_key: Optional[str] = None) -> dict:
    """Sync from Zotero and optionally embed new documents."""
    return await get_server().sync_and_embed(embed_sentences=embed_sentences, document_key=document_key)


@mcp.tool
async def get_embedding_status() -> dict:
    """Get current embedding statistics."""
    return await get_server().get_embedding_status()


@mcp.tool
async def delete_document(document_key: str) -> dict:
    """Delete all embeddings for a document."""
    return await get_server().delete_document(document_key)


@mcp.tool
async def reembed_document(document_key: str) -> dict:
    """Re-embed a specific document."""
    return await get_server().reembed_document(document_key)


@mcp.tool
async def import_item_by_doi(doi: str, collection_key: Optional[str] = None) -> dict:
    """Fetch BibTeX for a DOI and import the item into Zotero via the Connector."""
    return await get_server().import_item_by_doi(doi=doi, collection_key=collection_key)


def main() -> None:
    """Run the FastMCP server (stdio)."""
    logging.basicConfig(level=logging.INFO)

    config = Config()
    server = MCPZoteroServer(config)
    set_server_instance(server)

    logger.info("FastMCP server starting with stdio transport...")

    # Note: background embedding is started by main.py in the full app.
    # Keeping this entrypoint minimal for standalone use.
    mcp.run()


if __name__ == "__main__":
    main()
