import sys
import os
import asyncio
import threading

import pytest

# Add src to path like main.py does
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from semtero.config import Config
from semtero.mcp_server import MCPZoteroServer
from semtero.models import SearchResult, EmbeddingStatus, Document


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
    server.do_reranking = lambda results, query: results

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


def test_shutdown_closes_http_clients(monkeypatch):
    config = Config()
    server = MCPZoteroServer(config)

    calls = []

    monkeypatch.setattr(
        server.embedding_manager, "shutdown", lambda: calls.append("embedding")
    )
    monkeypatch.setattr(
        server.zotero_client.session, "close", lambda: calls.append("zotero")
    )
    monkeypatch.setattr(server.doi_client.session, "close", lambda: calls.append("doi"))

    server.shutdown()

    assert calls == ["embedding", "zotero", "doi"]


def test_server_shares_vector_store_between_embedding_and_search():
    config = Config()
    server = MCPZoteroServer(config)

    assert server.embedding_manager.vector_store is server.search_engine.vector_store


def test_do_reranking_passes_vram_threshold_and_batch_size_and_releases_device(monkeypatch):
    config = Config()
    config.RERANKER_GPU_MIN_VRAM_GB = 12.0
    config.RERANKER_BATCH_SIZE = 4
    server = MCPZoteroServer(config)

    calls = []

    class FakeReranker:
        def __init__(self, model_name=None, min_gpu_vram_gb=8.0, batch_size=8):
            calls.append(("init", min_gpu_vram_gb, batch_size))

        def rerank(self, results, query):
            calls.append(("rerank", query, len(results)))
            return list(reversed(results))

        def release_device(self):
            calls.append(("release",))

    monkeypatch.setattr("semtero.mcp_server.Reranker", FakeReranker)

    results = [
        SearchResult(
            text="first",
            document_title="",
            section_title="",
            zotero_key="doc1",
            relevance_score=0.9,
        ),
        SearchResult(
            text="second",
            document_title="",
            section_title="",
            zotero_key="doc2",
            relevance_score=0.8,
        ),
    ]

    reranked = server.do_reranking(results, "query")

    assert [item[0] for item in calls] == ["init", "rerank", "release"]
    assert calls[0] == ("init", 12.0, 4)
    assert reranked[0].text == "second"


def test_do_reranking_releases_device_when_rerank_fails(monkeypatch):
    config = Config()
    config.RERANKER_BATCH_SIZE = 6
    server = MCPZoteroServer(config)

    calls = []

    class FakeReranker:
        def __init__(self, model_name=None, min_gpu_vram_gb=8.0, batch_size=8):
            calls.append(("init", min_gpu_vram_gb, batch_size))

        def rerank(self, results, query):
            calls.append(("rerank", query))
            raise RuntimeError("boom")

        def release_device(self):
            calls.append(("release",))

    monkeypatch.setattr("semtero.mcp_server.Reranker", FakeReranker)

    with pytest.raises(RuntimeError, match="boom"):
        server.do_reranking([], "query")

    assert [item[0] for item in calls] == ["init", "rerank", "release"]
    assert calls[0] == ("init", config.RERANKER_GPU_MIN_VRAM_GB, 6)


def test_mcp_search_documents_emits_progress_updates(monkeypatch):
    config = Config()
    server = MCPZoteroServer(config)

    monkeypatch.setattr(
        server,
        "_get_metadata_for_key",
        lambda key: {"title": "Doc Title", "bibtex": "", "file_path": "", "authors": [], "date": "", "item_type": ""},
    )
    server.search_engine.vector_store.get_document_title = lambda key: "Doc Title"
    server.do_reranking = lambda results, query: results
    server.search_engine.search_best_sentences = lambda **kwargs: [
        SearchResult(
            text="match",
            document_title="",
            section_title="",
            zotero_key="doc1",
            relevance_score=0.9,
        )
    ]

    updates = []
    out = asyncio.run(
        server.search_documents(
            query="query",
            min_relevance=0.0,
            progress_callback=updates.append,
        )
    )

    assert len(out) == 1
    assert any(update.get("stage") == "reranking" for update in updates)
    assert any(
        update.get("message") == "Gathering Final Metadata 1 of 1"
        for update in updates
    )
    assert updates[-1]["stage"] == "complete"
    assert updates[-1]["result_count"] == 1


def test_get_pending_documents_skips_store_lookup_for_docs_already_in_metadata(
    monkeypatch,
):
    config = Config()
    server = MCPZoteroServer(config)

    docs = [
        Document(zotero_key="embedded", title="Already Embedded"),
        Document(zotero_key="missing", title="Needs Embedding"),
    ]

    monkeypatch.setattr(server.zotero_client, "get_documents_with_pdfs", lambda: iter(docs))
    monkeypatch.setattr(
        server.embedding_manager.vector_store,
        "get_embedded_documents",
        lambda: {"embedded": 12},
    )

    checked = []

    def fake_is_document_embedded(key):
        checked.append(key)
        return False

    monkeypatch.setattr(
        server.embedding_manager.vector_store,
        "is_document_embedded",
        fake_is_document_embedded,
    )

    pending, total_available = server._get_pending_documents()

    assert total_available == 2
    assert [doc.zotero_key for doc in pending] == ["missing"]
    assert checked == ["missing"]


def test_get_pending_documents_treats_zero_sentence_metadata_as_processed(
    monkeypatch,
):
    config = Config()
    server = MCPZoteroServer(config)

    docs = [
        Document(zotero_key="empty", title="Previously Empty"),
        Document(zotero_key="missing", title="Needs Embedding"),
    ]

    monkeypatch.setattr(server.zotero_client, "get_documents_with_pdfs", lambda: iter(docs))
    monkeypatch.setattr(
        server.embedding_manager.vector_store,
        "get_embedded_documents",
        lambda: {"empty": 0},
    )

    checked = []

    def fake_is_document_embedded(key):
        checked.append(key)
        return False

    monkeypatch.setattr(
        server.embedding_manager.vector_store,
        "is_document_embedded",
        fake_is_document_embedded,
    )

    pending, total_available = server._get_pending_documents()

    assert total_available == 2
    assert [doc.zotero_key for doc in pending] == ["missing"]
    assert checked == ["missing"]


def test_start_background_embedding_counts_skipped_documents_in_progress(monkeypatch):
    config = Config()
    server = MCPZoteroServer(config)

    submitted = []

    docs = [
        Document(zotero_key="done", title="Done"),
        Document(zotero_key="todo", title="Todo"),
        Document(zotero_key="empty", title="Previously Empty"),
    ]

    class FakeFuture:
        def result(self):
            return None

    monkeypatch.setattr(
        server.zotero_client,
        "get_documents_with_pdfs",
        lambda: iter(docs),
    )
    monkeypatch.setattr(
        server.embedding_manager.vector_store,
        "get_embedded_documents",
        lambda: {"done": 8, "empty": 0},
    )
    monkeypatch.setattr(
        server.embedding_manager.vector_store,
        "is_document_embedded",
        lambda key: False,
    )

    def fake_embed_document_async_with_client(doc, zotero_client, callback=None):
        submitted.append(doc.zotero_key)
        snapshot = server.embedding_manager.mark_document_completed(
            embedded_sentences=4 if doc.zotero_key == "todo" else 0
        )
        if callback:
            callback(snapshot)
        return FakeFuture()

    monkeypatch.setattr(
        server.embedding_manager,
        "embed_document_async_with_client",
        fake_embed_document_async_with_client,
    )

    server._run_background_embedding(trigger="manual")

    final_status = server.embedding_manager.get_embedding_status()
    assert final_status.is_running is False
    assert final_status.total_documents == 3
    assert final_status.processed_documents == 3
    assert submitted == ["todo"]


def test_start_background_embedding_begins_work_before_full_scan_finishes(monkeypatch):
    config = Config()
    server = MCPZoteroServer(config)

    first_doc_seen = threading.Event()
    allow_scan_to_finish = threading.Event()
    submitted = []

    docs = [
        Document(zotero_key="doc-1", title="Doc 1"),
        Document(zotero_key="doc-2", title="Doc 2"),
    ]

    def slow_docs():
        yield docs[0]
        first_doc_seen.set()
        assert allow_scan_to_finish.wait(timeout=2)
        yield docs[1]

    class FakeFuture:
        def result(self):
            return None

    monkeypatch.setattr(server.zotero_client, "get_documents_with_pdfs", slow_docs)
    monkeypatch.setattr(
        server.embedding_manager.vector_store,
        "get_embedded_documents",
        lambda: {},
    )
    monkeypatch.setattr(
        server.embedding_manager.vector_store,
        "is_document_embedded",
        lambda key: False,
    )

    def fake_embed_document_async_with_client(doc, zotero_client, callback=None):
        submitted.append(doc.zotero_key)
        return FakeFuture()

    monkeypatch.setattr(
        server.embedding_manager,
        "embed_document_async_with_client",
        fake_embed_document_async_with_client,
    )

    result = server.start_background_embedding(trigger="startup")
    assert result["status"] == "started"
    assert first_doc_seen.wait(timeout=1)

    status_during_scan = server.embedding_manager.get_embedding_status()
    assert status_during_scan.is_running is True
    assert status_during_scan.total_documents >= 1
    assert submitted == ["doc-1"]

    allow_scan_to_finish.set()
    assert server._embedding_thread is not None
    server._embedding_thread.join(timeout=2)

    final_status = server.embedding_manager.get_embedding_status()
    assert final_status.is_running is False
    assert final_status.total_documents == 2
    assert submitted == ["doc-1", "doc-2"]
