#!/usr/bin/env python3
"""Web UI for SemTero search.

Provides a web interface to search through embedded Zotero documents.
Run this alongside main.py or import and call run() to start the server.
"""

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any
from asyncio import run as asyncio_run

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flask import Flask, jsonify, render_template, request

# Import the MCP server components - we need direct access to search
from semtero.logging_setup import setup_logging
from semtero.mcp_server import get_server


logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

_SEARCH_PROGRESS_LOCK = threading.Lock()
_SEARCH_PROGRESS: dict[str, dict[str, Any]] = {}
_SEARCH_PROGRESS_TTL_SECONDS = 15 * 60


def _prune_search_progress_locked(now: float) -> None:
    cutoff = now - _SEARCH_PROGRESS_TTL_SECONDS
    stale_ids = [
        search_id
        for search_id, state in _SEARCH_PROGRESS.items()
        if state.get("finished") and state.get("updated_at", now) < cutoff
    ]
    for search_id in stale_ids:
        _SEARCH_PROGRESS.pop(search_id, None)


def _start_search_progress(search_id: str, query: str) -> None:
    now = time.time()
    with _SEARCH_PROGRESS_LOCK:
        _prune_search_progress_locked(now)
        _SEARCH_PROGRESS[search_id] = {
            "search_id": search_id,
            "query": query,
            "stage": "queued",
            "percentage": 4,
            "message": "Preparing search",
            "detail": "Starting semantic search",
            "finished": False,
            "error": "",
            "result_count": None,
            "updated_at": now,
        }


def _update_search_progress(search_id: str, **updates: Any) -> None:
    now = time.time()
    with _SEARCH_PROGRESS_LOCK:
        _prune_search_progress_locked(now)
        state = _SEARCH_PROGRESS.get(search_id)
        if not state:
            return
        if "percentage" in updates and updates["percentage"] is not None:
            updates["percentage"] = max(
                state.get("percentage", 0),
                max(0.0, min(100.0, float(updates["percentage"]))),
            )
        state.update(updates)
        state["updated_at"] = now


def _get_search_progress(search_id: str) -> dict[str, Any] | None:
    now = time.time()
    with _SEARCH_PROGRESS_LOCK:
        _prune_search_progress_locked(now)
        state = _SEARCH_PROGRESS.get(search_id)
        return dict(state) if state else None


@app.route("/")
def index():
    """Render the main search page."""
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def search():
    """
    Search API endpoint.
    
    Expects JSON body with:
        - query: str - The search query
        - top_sentences: int (optional) - Number of results to return, default 10
        - min_relevance: float (optional) - Minimum relevance score, default 0.75
    
    Returns JSON with results array.
    """
    data = request.get_json() or {}
    
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    top_sentences = int(data.get("top_sentences", 10))
    min_relevance = float(data.get("min_relevance", 0.75))
    require_cited_bibtex = data.get("require_cited_bibtex", False)
    search_id = str(data.get("search_id", "")).strip()

    if search_id:
        _start_search_progress(search_id, query)

    def progress_callback(progress: dict[str, Any]) -> None:
        if search_id:
            _update_search_progress(search_id, **progress)

    try:
        # Get the server instance and call search directly
        server = get_server()

        # Run the async search function in an event loop
        results = asyncio_run(server.search_documents(
            query=query,
            top_sentences=top_sentences,
            min_relevance=min_relevance,
            citation_return_mode="both",  # Get both text and citations
            require_cited_bibtex=require_cited_bibtex,
            progress_callback=progress_callback if search_id else None,
        ))

        if search_id:
            _update_search_progress(
                search_id,
                stage="complete",
                percentage=100,
                message=f"Found {len(results)} result{'s' if len(results) != 1 else ''}" if results else "No results found",
                detail="Search complete",
                finished=True,
                error="",
                result_count=len(results),
            )

        return jsonify({
            "results": results,
            "count": len(results),
            "query": query,
        })
        
    except RuntimeError as e:
        if search_id:
            _update_search_progress(
                search_id,
                stage="error",
                percentage=100,
                message="Search failed",
                detail=str(e),
                finished=True,
                error=str(e),
            )
        if "not registered" in str(e):
            return jsonify({"error": "MCP server not initialized. Please start main.py first."}), 503
        raise
    except Exception as e:
        logger.exception("Search error")
        if search_id:
            _update_search_progress(
                search_id,
                stage="error",
                percentage=100,
                message="Search failed",
                detail=str(e),
                finished=True,
                error=str(e),
            )
        return jsonify({"error": str(e)}), 500


@app.route("/api/search-progress/<search_id>", methods=["GET"])
def search_progress(search_id: str):
    """Get the current search progress for an in-flight or recent search."""
    progress = _get_search_progress(search_id)
    if not progress:
        return jsonify({"error": "Search progress not found"}), 404
    return jsonify(progress)


@app.route("/api/embed", methods=["POST"])
def trigger_embed():
    """Trigger a background pass to embed any newly discovered documents."""
    try:
        server = get_server()

        result = asyncio_run(server.embed_new_documents_now())
        status_code = 202 if result.get("status") == "started" else 200
        return jsonify(result), status_code

    except RuntimeError as e:
        if "not registered" in str(e):
            return jsonify({"error": "MCP server not initialized"}), 503
        raise
    except Exception as e:
        logger.exception("Embed trigger error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def status():
    """Get the embedding/collection status."""
    try:
        server = get_server()

        status_data = asyncio_run(server.get_embedding_status())

        return jsonify(status_data)
        
    except RuntimeError as e:
        if "not registered" in str(e):
            return jsonify({"error": "MCP server not initialized"}), 503
        raise
    except Exception as e:
        logger.exception("Status error")
        return jsonify({"error": str(e)}), 500


def run(host: str = "127.0.0.1", port: int = 23121, debug: bool = False):
    """Run the Flask web UI server."""
    setup_logging()
    logger.info(f"Starting SemTero Web UI on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run()