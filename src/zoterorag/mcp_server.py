"""MCP server for ZoteroRAG."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .config import Config
from .zotero_client import ZoteroClient
from .embedding_manager import EmbeddingManager
from .search_engine import SearchEngine


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
        if item_key in self._metadata_cache:
            return self._metadata_cache[item_key]
        
        # Try to get from Zotero
        metadata = self.zotero_client.get_item_metadata(item_key)
        if metadata:
            self._metadata_cache[item_key] = metadata
            return metadata
        
        # Return empty metadata structure
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
        top_sections: int = 5,
        top_sentences_per_section: int = 3
    ) -> list[dict]:
        """Search across embedded documents using two-stage RAG.
        
        Returns enriched results including BibTeX and file metadata.
        """
        results = self.search_engine.search(
            query=query,
            document_key=document_key,
            top_sections=top_sections,
            top_sentences_per_section=top_sentences_per_section
        )
        
        # Enrich results with metadata from Zotero or fallback to vector store titles
        enriched_results = []
        seen_keys = set()
        
        for r in results:
            result_dict = r.to_dict()
            
            # Get document-level metadata (only once per document)
            zotero_key = r.zotero_key
            
            if not zotero_key and result_dict.get("section_title"):
                # Try to extract document key from section - fallback to using
                # the title from vector store as a fallback for document_title
                pass
            
            if zotero_key and zotero_key not in seen_keys:
                seen_keys.add(zotero_key)
                
                # First try Zotero API metadata
                meta = self._get_metadata_for_key(zotero_key)
                
                # If no Zotero data, fall back to vector store title
                if not meta.get("title"):
                    vs_title = self.search_engine.vector_store.get_document_title(zotero_key)
                    if vs_title:
                        meta["title"] = vs_title
                
                # Update the result with enriched data
                result_dict["bibtex"] = meta.get("bibtex", "")
                result_dict["file_path"] = meta.get("file_path", "")
                result_dict["authors"] = meta.get("authors", [])
                result_dict["date"] = meta.get("date", "")
                result_dict["item_type"] = meta.get("item_type", "")
                
                # Also update document_title if empty - prefer Zotero title, fallback to vector store
                doc_title = meta.get("title", "")
                if not doc_title:
                    vs_title = self.search_engine.vector_store.get_document_title(zotero_key)
                    if vs_title:
                        doc_title = vs_title
                result_dict["document_title"] = doc_title
            
            # If we still don't have a document title, try to get it from section info
            if not result_dict.get("document_title") and result_dict.get("section_title"):
                # Use the section title's prefix as a fallback identifier
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

    def shutdown(self):
        """Shutdown the server."""
        self.embedding_manager.shutdown()


# --- MCP Protocol Helpers ---

def create_mcp_response(tool_name: str, result: Any) -> dict:
    """Create a standardized MCP response."""
    return {
        "jsonrpc": "2.0",
        "id": None,
        "result": {
            "tool": tool_name,
            "output": result
        }
    }


def create_mcp_error(code: int, message: str) -> dict:
    """Create a standardized MCP error response."""
    return {
        "jsonrpc": "2.0",
        "id": None,
        "error": {
            "code": code,
            "message": message
        }
    }


# --- Main entry point ---

def main():
    """Run the MCP server (standalone)."""
    logging.basicConfig(level=logging.INFO)
    
    config = Config()
    server = MCPZoteroServer(config)

    logger.info("MCP server starting on port 23120...")
    
    try:
        import asyncio
        # Start auto-embedding in background (non-blocking)
        asyncio.get_event_loop().run_until_complete(server.start_auto_embedding())
        
        # Keep running
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
