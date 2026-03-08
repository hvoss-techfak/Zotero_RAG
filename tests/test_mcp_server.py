import sys
import os
import asyncio

# Add src to path like main.py does
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from zoterorag.config import Config
from zoterorag.mcp_server import MCPZoteroServer
from zoterorag.models import SearchResult, EmbeddingStatus


def test_mcp_search_documents_passes_citation_return_mode_and_filters_require_cited_bibtex(
    monkeypatch,
):
    config = Config()
    server = MCPZoteroServer(config)

    monkeypatch.setattr(
        server, "_get_metadata_for_key", lambda key: {"title": "", "bibtex": ""}
    )
    server.search_engine.vector_store.get_document_title = lambda key: None

    r_with = SearchResult(
        text="hello",
        document_title="",
        section_title="",
        zotero_key="docA",
        relevance_score=0.9,
        rerank_score=0.9,
        cited_bibtex=["@article{a, title={A}}"],
    )
    r_without = SearchResult(
        text="world",
        document_title="",
        section_title="",
        zotero_key="docB",
        relevance_score=0.9,
        rerank_score=0.9,
        cited_bibtex=[],
    )

    calls = {}

    def fake_search_best_sentences(**kwargs):
        calls.update(kwargs)
        return [r_with, r_without]

    server.search_engine.search_best_sentences = fake_search_best_sentences

    out = asyncio.run(
        server.search_documents(
            query="q",
            citation_return_mode="bibtex",
            require_cited_bibtex=True,
            min_relevance=0.0,
        )
    )

    assert calls.get("citation_return_mode") == "bibtex"
    assert len(out) == 1
    assert out[0]["zotero_key"] == "docA"


def test_mcp_search_documents_enriches_all_results_with_document_bibtex(monkeypatch):
    config = Config()
    server = MCPZoteroServer(config)

    metadata = {
        "title": "Document Title",
        "bibtex": "@article{Doe2024Doc, title={Document Title}}",
        "file_path": "/tmp/doc.pdf",
        "authors": ["Jane Doe"],
        "date": "2024",
        "item_type": "journalArticle",
    }
    monkeypatch.setattr(server, "_get_metadata_for_key", lambda key: metadata)
    server.search_engine.vector_store.get_document_title = lambda key: None
    server.do_reranking = lambda results, query: results
    server.search_engine.search_best_sentences = lambda **kwargs: [
        SearchResult(
            text="match",
            document_title="",
            section_title="",
            zotero_key="doc1",
            relevance_score=0.9,
        ),
        SearchResult(
            text="match2",
            document_title="",
            section_title="",
            zotero_key="doc1",
            relevance_score=0.8,
        ),
    ]

    out = asyncio.run(server.search_documents(query="query", min_relevance=0.0))

    assert len(out) == 2
    assert all(result["bibtex"].startswith("@article") for result in out)
    assert all(result["document_title"] == "Document Title" for result in out)


def test_embed_new_documents_now_returns_started(monkeypatch):
    config = Config()
    server = MCPZoteroServer(config)

    started = []

    def fake_start(trigger="manual"):
        started.append(trigger)
        return {"status": "started", "trigger": trigger}

    monkeypatch.setattr(server, "start_background_embedding", fake_start)

    result = asyncio.run(server.embed_new_documents_now())

    assert result["status"] == "started"
    assert started == ["manual"]


def test_get_embedding_status_returns_progress_payload(monkeypatch):
    config = Config()
    server = MCPZoteroServer(config)

    monkeypatch.setattr(
        server.search_engine,
        "get_stats",
        lambda: {"total_sentences": 12, "embedded_documents": 2},
    )
    monkeypatch.setattr(
        server.embedding_manager.vector_store,
        "get_embedded_documents",
        lambda: {"A": 3, "B": 4},
    )
    monkeypatch.setattr(
        server.embedding_manager,
        "get_embedding_status",
        lambda: EmbeddingStatus(
            total_documents=4,
            processed_documents=2,
            embedded_sentences=50,
            is_running=True,
        ),
    )
    server.set_next_auto_reembed_at("2026-03-08T10:15:00+00:00")

    payload = asyncio.run(server.get_embedding_status())

    assert payload["total_sentences"] == 12
    assert payload["embedded_documents"] == 2
    assert payload["processed_documents"] == 2
    assert payload["next_auto_reembed_at"] == "2026-03-08T10:15:00+00:00"
