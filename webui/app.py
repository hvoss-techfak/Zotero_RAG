#!/usr/bin/env python3
"""Web UI for ZoteroRAG search.

Provides a web interface to search through embedded Zotero documents.
Run this alongside main.py or import and call run() to start the server.
"""

import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flask import Flask, jsonify, render_template, request

# Import the MCP server components - we need direct access to search
from zoterorag.logging_setup import setup_logging
from zoterorag.mcp_server import get_server


logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)


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
    
    try:
        # Get the server instance and call search directly
        server = get_server()
        
        import asyncio
        
        # Run the async search function in an event loop
        results = asyncio.run(server.search_documents(
            query=query,
            top_sentences=top_sentences,
            min_relevance=min_relevance,
            citation_return_mode="both",  # Get both text and citations
            require_cited_bibtex=require_cited_bibtex,
        ))
        
        return jsonify({
            "results": results,
            "count": len(results),
            "query": query,
        })
        
    except RuntimeError as e:
        if "not registered" in str(e):
            return jsonify({"error": "MCP server not initialized. Please start main.py first."}), 503
        raise
    except Exception as e:
        logger.exception("Search error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/embed", methods=["POST"])
def trigger_embed():
    """Trigger a background pass to embed any newly discovered documents."""
    try:
        server = get_server()

        import asyncio
        result = asyncio.run(server.embed_new_documents_now())
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
        
        import asyncio
        status_data = asyncio.run(server.get_embedding_status())
        
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
    logger.info(f"Starting ZoteroRAG Web UI on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run()