#!/usr/bin/env python3
"""ZoteroRAG - Main entry point for MCP server.

Usage:
    python main.py                 # Run as MCP server on port 23120
    python main.py --daemon       # Run in daemon mode with auto-sync
    python main.py --test-ollama  # Test Ollama connectivity

Environment variables:
    ZOTERO_API_URL        - Zotero local API URL (default: http://127.0.0.1:23119)
    OLLAMA_BASE_URL       - Ollama server URL (default: http://localhost:11434)
    VECTOR_STORE_DIR     - Vector store directory
    PDF_CACHE_PATH       - PDF cache directory
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import asyncio

from zoterorag.config import Config
from zoterorag.embedding_manager import EmbeddingManager
from zoterorag.mcp_server import MCPZoteroServer, set_server_instance, mcp
from zoterorag.search_engine import SearchEngine
from zoterorag.vector_store import VectorStore
from zoterorag.zotero_client import ZoteroClient


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ZoteroRAGApplication:
    """Main application managing all ZoteroRAG services."""

    def __init__(self):
        self.config = Config()
        self.server: MCPZoteroServer | None = None
        self.embedding_manager: EmbeddingManager | None = None
        self._running = False
        self._daemon_thread: threading.Thread | None = None

    def initialize(self) -> bool:
        """Initialize all services."""
        logger.info("Initializing ZoteroRAG...")

        # Ensure directories exist
        Config.ensure_dirs()

        # Initialize components - let MCPZoteroServer handle initialization properly
        try:
            zotero_client = ZoteroClient(api_url=self.config.ZOTERO_API_URL)
            
            # Create and register server instance for global access
            # This handles proper EmbeddingManager initialization with executor
            self.server = MCPZoteroServer(self.config)
            set_server_instance(self.server)  # Register for FastMCP tool functions
            
            # Reference the embedding_manager from the server (already properly initialized)
            self.embedding_manager = self.server.embedding_manager

            logger.info("All services initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False

    def run_mcp_server(self):
        """Run the MCP server (blocking)."""
        # Initialize if not already done
        if not self.server:
            if not self.initialize():
                sys.exit(1)

        logger.info("Testing service connections...")
        
        zotero_ok = self.test_zotero_connection()
        ollama_ok = self.test_ollama_connection()

        if not zotero_ok:
            logger.warning("Zotero connection failed - some features may not work")

        if not ollama_ok:
            logger.error("Ollama connection required for embeddings")
            sys.exit(1)

        # Run the MCP server (this is blocking)
        logger.info(f"MCP server starting with stdio transport...")
        
        try:
            # Use mcp.run() directly - it handles its own event loop
            import anyio
            anyio.run(mcp.run_stdio_async)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def run_daemon(self):
        """Run in daemon mode with periodic sync."""
        if not self.server:
            if not self.initialize():
                sys.exit(1)

        logger.info("Running in daemon mode...")
        
        while self._running:
            try:
                from zoterorag.mcp_server import get_server
                server = get_server()
                
                async def check_sync():
                    result = await server.sync_and_embed(embed_sentences=False)
                    logger.info(f"Sync status: {result}")
                    
                asyncio.run(check_sync())
                
            except Exception as e:
                logger.error(f"Daemon error: {e}")

            time.sleep(60)  # Check every minute

    def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down...")
        
        if self.embedding_manager:
            self.embedding_manager.shutdown()
            
        if self.server:
            self.server.shutdown()
            
        logger.info("Shutdown complete")

    def test_zotero_connection(self) -> bool | str:
        """Test Zotero API connectivity.
        
        Returns:
            True if connection works,
            "no_server" if Zotero isn't running,
            False for other failures
        """
        try:
            client = ZoteroClient(api_url=self.config.ZOTERO_API_URL)
            return client.check_connection()
        except Exception as e:
            logger.debug(f"Zotero connection test failed: {e}")
            # Check if it's a connection error
            if "Connection" in str(e) or "refused" in str(e).lower():
                return "no_server"
            return False

    def test_ollama_connection(self) -> bool | str:
        """Test Ollama connectivity and model availability.
        
        Returns:
            True if embedding works,
            "no_server" if Ollama isn't running,  
            "no_model" if the model isn't available,
            False for other failures
        """
        try:
            # Initialize EmbeddingManager which uses Ollama
            with patch("zoterorag.vector_store.VectorStore"):
                manager = EmbeddingManager(self.config)
            
            # Try to embed a simple test text
            embedding = manager.embed_text("test")
            
            if embedding and len(embedding) > 0:
                return True
            
            return False
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.debug(f"Ollama connection test failed: {e}")
            
            # Determine specific failure reason
            if "connection" in error_msg or "refused" in error_msg or "connect" in error_msg:
                return "no_server"
            elif "model" in error_msg or "not found" in error_msg or "404" in error_msg:
                return "no_model"
            
            return False

    def signal_handler(self, signum, frame):
        """Handle termination signals."""
        self._running = False
        self.shutdown()
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="ZoteroRAG - MCP Server for Zotero with RAG"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run in daemon mode with periodic sync"
    )
    parser.add_argument(
        "--test-ollama",
        action="store_true",
        help="Test Ollama connectivity and exit"
    )
    parser.add_argument(
        "--test-zotero",
        action="store_true", 
        help="Test Zotero API connectivity and exit"
    )
    args = parser.parse_args()

    app = ZoteroRAGApplication()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, app.signal_handler)
    signal.signal(signal.SIGTERM, app.signal_handler)

    if args.test_ollama:
        Config.ensure_dirs()
        result = app.test_ollama_connection()
        if result is True:
            print("✓ Ollama embedding model works")
            sys.exit(0)
        elif result == "no_server":
            print(f"✗ No Ollama server running at {Config.OLLAMA_BASE_URL}")
            sys.exit(1)
        elif result == "no_model":
            print(f"✗ Model '{Config.EMBEDDING_MODEL}' not found in Ollama")
            sys.exit(1)
        else:
            print("✗ Ollama connection failed")
            sys.exit(1)

    if args.test_zotero:
        Config.ensure_dirs()
        result = app.test_zotero_connection()
        if result is True:
            print("✓ Zotero API connection OK")  
            sys.exit(0)
        elif result == "no_server":
            print("✗ No Zotero server running at {}. Make sure the Zotero local connector is running.".format(Config.ZOTERO_API_URL))
            sys.exit(1)
        else:
            print("✗ Zotero API connection failed")
            sys.exit(1)
    
    # Start running
    app._running = True
    
    try:
        if args.daemon:
            app.run_daemon()
        else:
            app.run_mcp_server()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        app.shutdown()


if __name__ == "__main__":
    main()