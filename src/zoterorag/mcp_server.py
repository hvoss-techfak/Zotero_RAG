"""MCP server for ZoteroRAG."""

import logging
from typing import Optional

from fastmcp import FastMCP

from .config import Config
from .embedding_manager import EmbeddingManager
from .search_engine import SearchEngine
from .zotero_client import ZoteroClient


logger = logging.getLogger(__name__)


class MCPZoteroServer:
    """MCP server exposing Zotero RAG tools."""

    def __init__(self, config: Config):
        self.config = config
        self.zotero_client = ZoteroClient(api_url=config.ZOTERO_API_URL)
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
        print(f"[DEBUG] _get_metadata_for_key called with key: {item_key}")
        
        if item_key in self._metadata_cache:
            print(f"[DEBUG] Found metadata in cache for key: {item_key}")
            return self._metadata_cache[item_key]
        
        # Try to get from Zotero
        print(f"[DEBUG] Calling zotero_client.get_item_metadata({item_key})")
        metadata = self.zotero_client.get_item_metadata(item_key)
        print(f"[DEBUG] Got metadata result: {metadata}")
        
        if metadata:
            self._metadata_cache[item_key] = metadata
            return metadata
        
        # Return empty metadata structure
        print(f"[DEBUG] No metadata found, returning empty structure for key: {item_key}")
        return {
            "bibtex": "",
            "file_path": "",
            "title": "",
            "authors": [],
            "date": "",
            "item_type": ""
        }

    # --- MCP Tool Implementations ---

    async def search_documents(
        self,
        query: str,
        document_key: Optional[str] = None,
        top_sentences: int = 10,
        min_relevance: float = 0.75,
    ) -> list[dict]:
        """Search across embedded documents using two-stage RAG.
        
        Returns enriched results including BibTeX and file metadata.
        """
        print(f"[DEBUG] search_documents called with query: {query}")
        
        results = self.search_engine.search(
            query=query,
            document_key=document_key,
            top_sentences=top_sentences,
        )
        
        print(f"[DEBUG] Got {len(results)} search results")
        
        # First pass: collect all unique keys and fetch metadata for each once
        # This ensures we call Zotero API at most once per unique document
        key_to_metadata: dict[str, dict] = {}
        
        for r in results:
            zotero_key = r.zotero_key
            print(f"[DEBUG] Processing result with zotero_key: {zotero_key}")
            
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

        ret = []
        # Remove results below threshold:
        for r in enriched_results:
            if r.get("relevance_score", 0) >= min_relevance:
                ret.append(r)
            else:
                print(f"[DEBUG] Filtering out result with relevance {r.get('relevance_score', 0)} below threshold {min_relevance}")
        
        return ret

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
    min_relevance: float = 0.75,
) -> list[dict]:
    print(f"[DEBUG] MCP tool search_documents called with query: {query}")
    """Search across embedded documents using two-stage RAG."""
    return await get_server().search_documents(
        query=query,
        document_key=document_key,
        top_sentences=top_sentences,
        min_relevance=min_relevance
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
