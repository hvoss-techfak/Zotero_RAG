#!/usr/bin/env python3
"""SemTero - main application entrypoint."""

import argparse
import asyncio
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from semtero.config import Config
from semtero.embedding_manager import EmbeddingManager
from semtero.logging_setup import setup_logging
from semtero.mcp_server import MCPZoteroServer, mcp, set_server_instance
from semtero.zotero_client import ZoteroClient

try:
    from webui.app import run as run_webui

    WEBUI_AVAILABLE = True
except ImportError:
    run_webui = None
    WEBUI_AVAILABLE = False


setup_logging()
logger = logging.getLogger(__name__)


class SemTeroApplication:
    """Main application managing all SemTero services."""

    def __init__(self) -> None:
        self.config = Config()
        self.server: MCPZoteroServer | None = None
        self.embedding_manager: EmbeddingManager | None = None
        self._running = False
        self._scheduler_thread: threading.Thread | None = None
        self._progress_bar_active = False
        self._last_embed_progress_log_ts = 0.0
        self._embed_progress_interval_sec = self.config.EMBED_PROGRESS_INTERVAL_SEC

    def initialize(self) -> bool:
        """Initialize all services."""
        logger.info("Initializing SemTero...")
        Config.ensure_dirs()

        try:
            self.server = MCPZoteroServer(self.config)
            set_server_instance(self.server)
            self.embedding_manager = self.server.embedding_manager
            self.server.register_embedding_status_listener(
                self._handle_embedding_status
            )
            logger.info("All services initialized successfully")
            return True
        except Exception as exc:
            logger.error("Failed to initialize: %s", exc)
            return False

    def start_background_embedding(self, trigger: str = "manual") -> dict:
        """Start background embedding for pending documents."""
        if not self.server:
            logger.warning("Cannot start embedding - services not initialized")
            return {"status": "error", "message": "Services not initialized"}
        return self.server.start_background_embedding(trigger=trigger)

    def _handle_embedding_status(self, status) -> None:
        self._emit_embed_progress(status=status, force=not status.is_running)

    @staticmethod
    def _format_embed_progress_line(status) -> str:
        total = max(0, status.total_documents)
        processed = (
            min(status.processed_documents, total)
            if total
            else status.processed_documents
        )
        pct = status.progress_percentage if total else 0.0
        width = 28
        filled = int((pct / 100.0) * width) if total else 0
        bar = "█" * filled + "░" * (width - filled)
        state = "RUNNING" if status.is_running else "DONE"
        return (
            f"[Embed {state}] [{bar}] {processed}/{total} docs "
            f"({pct:5.1f}%) | sentences={status.embedded_sentences} "
            f"| failed={status.failed_documents}"
        )

    def _emit_embed_progress(self, *, status=None, force: bool = False) -> None:
        """Emit a compact CLI progress bar for background embedding."""
        if status is None:
            if not self.embedding_manager:
                return
            status = self.embedding_manager.get_embedding_status()

        if status.total_documents <= 0 and not status.is_running:
            if self._progress_bar_active:
                sys.stderr.write("\n")
                sys.stderr.flush()
                self._progress_bar_active = False
            return

        now = time.time()
        if (
            not force
            and (now - self._last_embed_progress_log_ts)
            < self._embed_progress_interval_sec
        ):
            return

        self._last_embed_progress_log_ts = now
        line = self._format_embed_progress_line(status)
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            sys.stderr.write("\r" + line.ljust(140))
            if status.is_running:
                self._progress_bar_active = True
            else:
                sys.stderr.write("\n")
                self._progress_bar_active = False
            sys.stderr.flush()
            return

        print(line, file=sys.stderr)
        if not status.is_running:
            self._progress_bar_active = False

    def _start_auto_reembed_scheduler(self) -> None:
        """Auto-start a new embed scan after the configured cool-down."""
        if not self.server or self.config.AUTO_REEMBED_INTERVAL_MINUTES <= 0:
            return
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            return

        interval_minutes = self.config.AUTO_REEMBED_INTERVAL_MINUTES

        def run_scheduler() -> None:
            while self._running:
                try:
                    if not self.embedding_manager:
                        time.sleep(5)
                        continue

                    status = self.embedding_manager.get_embedding_status()
                    if status.is_running:
                        self.server.set_next_auto_reembed_at("")
                        time.sleep(5)
                        continue

                    if not status.finished_at:
                        self.server.set_next_auto_reembed_at("")
                        time.sleep(5)
                        continue

                    finished_at = datetime.fromisoformat(status.finished_at)
                    if finished_at.tzinfo is None:
                        finished_at = finished_at.replace(tzinfo=timezone.utc)

                    next_run = finished_at + timedelta(minutes=interval_minutes)
                    self.server.set_next_auto_reembed_at(
                        next_run.astimezone(timezone.utc)
                        .replace(microsecond=0)
                        .isoformat()
                    )

                    if datetime.now(timezone.utc) >= next_run.astimezone(timezone.utc):
                        result = self.server.start_background_embedding(
                            trigger="scheduled"
                        )
                        if result.get("status") == "started":
                            self.server.set_next_auto_reembed_at("")

                    time.sleep(5)
                except Exception as exc:
                    logger.warning("Auto re-embed scheduler error: %s", exc)
                    time.sleep(5)

        self._scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self._scheduler_thread.start()

    def _start_webui(self, *, port: int = 23121, host: str | None = None) -> None:
        """Start the Flask web UI in a separate thread."""
        if not WEBUI_AVAILABLE or run_webui is None:
            logger.warning("Web UI not available. Install Flask.")
            return

        webui_host = host or self.config.WEBUI_HOST
        try:
            thread = threading.Thread(
                target=run_webui,
                kwargs={"host": webui_host, "port": port, "debug": False},
                daemon=True,
            )
            thread.start()
            logger.info("Web UI started on http://%s:%s", webui_host, port)
        except Exception as exc:
            logger.error("Failed to start web UI: %s", exc)

    def run_mcp_server(
        self, *, enable_webui: bool = True, webui_port: int = 23121
    ) -> None:
        """Run the MCP server with stdio transport."""
        if not self.server and not self.initialize():
            sys.exit(1)

        logger.info("Testing service connections...")
        zotero_ok = self.test_zotero_connection()
        ollama_ok = self.test_ollama_connection()

        if not zotero_ok:
            logger.warning("Zotero connection failed - some features may not work")
        if not ollama_ok:
            logger.error("Ollama connection required for embeddings")
            sys.exit(1)

        self.start_background_embedding(trigger="startup")
        self._start_auto_reembed_scheduler()
        if enable_webui:
            self._start_webui(port=webui_port)

        try:
            import anyio

            async def _run_stdio() -> None:
                await mcp.run_stdio_async()

            anyio.run(_run_stdio)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def run_daemon(self) -> None:
        """Run in daemon mode with periodic sync."""
        if not self.server and not self.initialize():
            sys.exit(1)

        logger.info("Running in daemon mode...")
        while self._running:
            try:

                async def check_sync() -> None:
                    result = await self.server.sync_and_embed(embed_sentences=False)
                    logger.info("Sync status: %s", result)

                asyncio.run(check_sync())
            except Exception as exc:
                logger.error("Daemon error: %s", exc)
            time.sleep(60)

    def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")
        if self.server:
            self.server.shutdown()
        elif self.embedding_manager:
            self.embedding_manager.shutdown()
        logger.info("Shutdown complete")

    def test_zotero_connection(self) -> bool | str:
        """Test Zotero API connectivity."""
        client = None
        try:
            client = ZoteroClient(api_url=self.config.ZOTERO_API_URL)
            return client.check_connection()
        except Exception as exc:
            logger.debug("Zotero connection test failed: %s", exc)
            if "connection" in str(exc).lower() or "refused" in str(exc).lower():
                return "no_server"
            return False
        finally:
            if client is not None:
                client.close()

    def test_ollama_connection(self) -> bool | str:
        """Test Ollama connectivity and model availability."""
        try:
            with patch("semtero.vector_store.VectorStore"):
                manager = EmbeddingManager(self.config)
            embedding = manager.embed_text(["test"])
            return bool(embedding and len(embedding) > 0)
        except Exception as exc:
            error_msg = str(exc).lower()
            logger.debug("Ollama connection test failed: %s", exc)
            if (
                "connection" in error_msg
                or "refused" in error_msg
                or "connect" in error_msg
            ):
                return "no_server"
            if "model" in error_msg or "not found" in error_msg or "404" in error_msg:
                return "no_model"
            return False

    def signal_handler(self, signum, frame) -> None:
        """Handle termination signals."""
        self._running = False
        self.shutdown()
        sys.exit(0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SemTero - MCP Server for Zotero with RAG"
    )
    parser.add_argument(
        "--daemon", action="store_true", help="Run in daemon mode with periodic sync"
    )
    parser.add_argument(
        "--test-ollama", action="store_true", help="Test Ollama connectivity and exit"
    )
    parser.add_argument(
        "--test-zotero",
        action="store_true",
        help="Test Zotero API connectivity and exit",
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="sse",
        choices=["stdio", "http", "sse", "streamable-http"],
        help="Transport protocol to use (default: sse)",
    )
    parser.add_argument(
        "--port", type=int, default=23120, help="Port for HTTP/SSE transports"
    )
    parser.add_argument("--no-webui", action="store_true", help="Disable the web UI")
    parser.add_argument(
        "--webui-port", type=int, default=23121, help="Port for the web UI"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("APP_HOST", "127.0.0.1"),
        help="Bind host for HTTP/SSE transports (default: APP_HOST or 127.0.0.1)",
    )
    parser.add_argument(
        "--webui-host",
        type=str,
        default=os.getenv("WEBUI_HOST", os.getenv("APP_HOST", "127.0.0.1")),
        help="Bind host for the web UI (default: WEBUI_HOST or APP_HOST)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    app = SemTeroApplication()
    app.config.APP_HOST = args.host
    app.config.WEBUI_HOST = args.webui_host

    signal.signal(signal.SIGINT, app.signal_handler)
    signal.signal(signal.SIGTERM, app.signal_handler)

    if args.test_ollama:
        Config.ensure_dirs()
        result = app.test_ollama_connection()
        if result is True:
            print("✓ Ollama embedding model works")
            sys.exit(0)
        if result == "no_server":
            print(f"✗ No Ollama server running at {app.config.OLLAMA_BASE_URL}")
            sys.exit(1)
        if result == "no_model":
            print(f"✗ Model '{app.config.EMBEDDING_MODEL}' not found in Ollama")
            sys.exit(1)
        print("✗ Ollama connection failed")
        sys.exit(1)

    if args.test_zotero:
        Config.ensure_dirs()
        result = app.test_zotero_connection()
        if result is True:
            print("✓ Zotero API connection OK")
            sys.exit(0)
        if result == "no_server":
            print(
                "✗ No Zotero server running at {}. Make sure the Zotero local connector is running.".format(
                    app.config.ZOTERO_API_URL
                )
            )
            sys.exit(1)
        print("✗ Zotero API connection failed")
        sys.exit(1)

    app._running = True

    if args.transport != "stdio":
        if not app.initialize():
            sys.exit(1)

        logger.info("Testing service connections...")
        zotero_ok = app.test_zotero_connection()
        ollama_ok = app.test_ollama_connection()
        if not zotero_ok:
            logger.warning("Zotero connection failed - some features may not work")
        if not ollama_ok:
            logger.error("Ollama connection required for embeddings")
            sys.exit(1)

        app.start_background_embedding(trigger="startup")
        app._start_auto_reembed_scheduler()
        if not args.no_webui:
            app._start_webui(host=args.webui_host, port=args.webui_port)

        logger.info(
            "MCP server starting with %s transport on port %s...",
            args.transport,
            args.port,
        )
        try:
            mcp.run(transport=args.transport, host=args.host, port=args.port)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            app.shutdown()
        return

    if not WEBUI_AVAILABLE and not args.no_webui:
        logger.warning("Web UI not available. Install Flask: uv add flask")

    try:
        if args.daemon:
            app.run_daemon()
        else:
            app.run_mcp_server(
                enable_webui=not args.no_webui, webui_port=args.webui_port
            )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        app.shutdown()


if __name__ == "__main__":
    main()
