"""Tests for skipping already-embedded documents.

We want idempotent behaviour: if a document key is already marked as embedded
in embedded_docs.json, embedding should be skipped unless explicitly re-embedded.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from zoterorag.config import Config
from zoterorag.embedding_manager import EmbeddingManager
from zoterorag.models import Document


def test_embed_document_async_with_client_skips_if_already_embedded(
    tmp_path: Path,
) -> None:
    config = Config()
    config.VECTOR_STORE_DIR = str(tmp_path)

    mgr = EmbeddingManager(config)

    # Start a job so processed_documents is meaningful
    mgr.start_embedding_job(total_documents=1)

    # Pretend this doc was already embedded earlier.
    mgr.vector_store.update_embedded_document("D1", 5)

    doc = Document(zotero_key="D1", title="Already embedded")

    zotero_client = MagicMock()

    fut = mgr.embed_document_async_with_client(doc, zotero_client)
    fut.result(timeout=2)

    # If we skipped properly, we never tried to fetch the PDF.
    assert zotero_client.get_pdf_bytes.call_count == 0
    assert zotero_client.get_group_pdf_bytes.call_count == 0

    # But it should still count as processed for the active job.
    assert mgr.get_embedding_status().processed_documents == 1


def test_embed_document_async_with_client_skips_if_previously_processed_with_zero_sentences(
    tmp_path: Path,
) -> None:
    config = Config()
    config.VECTOR_STORE_DIR = str(tmp_path)

    mgr = EmbeddingManager(config)
    mgr.start_embedding_job(total_documents=1)

    # A prior run found the PDF but extracted no sentences.
    mgr.vector_store.update_embedded_document("D0", 0)

    doc = Document(zotero_key="D0", title="Previously empty")
    zotero_client = MagicMock()

    fut = mgr.embed_document_async_with_client(doc, zotero_client)
    fut.result(timeout=2)

    assert zotero_client.get_pdf_bytes.call_count == 0
    assert zotero_client.get_group_pdf_bytes.call_count == 0
    assert mgr.get_embedding_status().processed_documents == 1


def test_embed_document_async_with_client_runs_if_not_embedded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Config()
    config.VECTOR_STORE_DIR = str(tmp_path)

    mgr = EmbeddingManager(config)

    doc = Document(zotero_key="D2", title="Not embedded")

    zotero_client = MagicMock()
    zotero_client.get_pdf_bytes.return_value = (
        b"%PDF-1.4 fake"  # won't be parsed; we stub below
    )

    # Avoid needing a real PDF by stubbing out the inner task early.
    called = {"n": 0}

    def fake_task(document, client, callback):
        called["n"] += 1
        return None

    monkeypatch.setattr(mgr, "_embed_document_from_zotero_task", fake_task)

    fut = mgr.embed_document_async_with_client(doc, zotero_client)
    fut.result(timeout=2)

    assert called["n"] == 1
