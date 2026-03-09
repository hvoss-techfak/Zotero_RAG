"""Tests for SearchEngine - sentence embedding search."""

import sys
import os
from unittest.mock import patch, MagicMock

import pytest

# Add src to path like main.py does
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from semtero.search_engine import SearchEngine
from semtero.config import Config
from semtero.models import SearchResult


class TestSearchEngineInitialization:
    """Test suite for SearchEngine initialization."""

    @patch("semtero.search_engine.VectorStore")
    def test_default_initialization(self, mock_vector_store):
        """Test default initialization."""
        config = Config()

        engine = SearchEngine(config)

        assert engine.config is not None
        assert engine.vector_store is not None


class TestQueryEmbedding:
    """Test suite for query embedding generation."""

    @patch("semtero.search_engine.Client")
    @patch("semtero.search_engine.VectorStore")
    def test_get_query_embedding(self, mock_vector_store, mock_client_cls):
        """Test query embedding generation."""
        mock_client = MagicMock()
        mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_client_cls.return_value = mock_client

        config = Config()
        config.EMBEDDING_DIMENSIONS = 0
        engine = SearchEngine(config)
        engine.vector_store.get_detected_dimension.return_value = None

        result = engine._get_query_embedding("test query")

        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.assert_called_once()

    @patch("semtero.search_engine.Client")
    @patch("semtero.search_engine.VectorStore")
    def test_get_query_embedding_raises_on_dimension_mismatch(
        self, mock_vector_store, mock_client_cls
    ):
        mock_client = MagicMock()
        mock_client.embeddings.return_value = {"embedding": [0.1] * 2560}
        mock_client_cls.return_value = mock_client

        config = Config()
        config.EMBEDDING_DIMENSIONS = 1024
        engine = SearchEngine(config)
        engine.vector_store.get_detected_dimension.return_value = None

        with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS=1024"):
            engine._get_query_embedding("test query")


class TestSearchBestSentences:
    """Search should use ANN ids+distances and only fetch top-k texts."""

    @patch("semtero.search_engine.Client")
    @patch("semtero.search_engine.VectorStore")
    def test_search_best_sentences_uses_ids_and_fetches_texts(
        self, mock_vector_store_cls, mock_client_cls
    ):
        mock_client = MagicMock()
        mock_client.embeddings.return_value = {"embedding": [0.0, 0.0, 1.0]}
        mock_client_cls.return_value = mock_client

        config = Config()
        config.EMBEDDING_DIMENSIONS = 0
        engine = SearchEngine(config)

        mock_vs = MagicMock()
        mock_vs.get_detected_dimension.return_value = None
        mock_vs.search_sentence_ids.return_value = (
            ["s1", "s2"],
            [0.25, 0.5],
            [{"document_key": "docA"}, {"document_key": "docB"}],
        )
        mock_vs.get_sentence_texts_by_ids.return_value = {"s1": "hello", "s2": "world"}
        engine.vector_store = mock_vs

        results = engine.search_best_sentences("q", top_sentences=2)

        assert [r.text for r in results] == ["world", "hello"]
        mock_vs.search_sentence_ids.assert_called_once()
        mock_vs.get_sentence_texts_by_ids.assert_called_once_with(["s1", "s2"])
        assert all(isinstance(r, SearchResult) for r in results)

    @patch("semtero.search_engine.Client")
    @patch("semtero.search_engine.VectorStore")
    def test_search_best_sentences_citation_return_modes(
        self, mock_vector_store_cls, mock_client_cls
    ):
        mock_client = MagicMock()
        mock_client.embeddings.return_value = {"embedding": [0.0, 0.0, 1.0]}
        mock_client_cls.return_value = mock_client

        config = Config()
        config.EMBEDDING_DIMENSIONS = 0
        engine = SearchEngine(config)

        mock_vs = MagicMock()
        mock_vs.get_detected_dimension.return_value = None
        mock_vs.search_sentence_ids.return_value = (
            ["s1"],
            [0.25],
            [
                {
                    "document_key": "docA",
                    "citation_numbers": [5],
                    "referenced_texts": ["A. Author. Some Paper. 2020."],
                    "referenced_bibtex": ["@article{author20205, title={Some Paper}}"],
                }
            ],
        )
        mock_vs.get_sentence_texts_by_ids.return_value = {"s1": "hello [5]"}
        engine.vector_store = mock_vs

        # sentence-only
        r1 = engine.search_best_sentences(
            "q", top_sentences=1, citation_return_mode="sentence"
        )
        assert r1[0].text == "hello [5]"
        assert r1[0].citation_numbers == [5]
        assert r1[0].cited_bibtex

        # bibtex-only
        r2 = engine.search_best_sentences(
            "q", top_sentences=1, citation_return_mode="bibtex"
        )
        assert "@article" in r2[0].text

        # both
        r3 = engine.search_best_sentences(
            "q", top_sentences=1, citation_return_mode="both"
        )
        assert "hello [5]" in r3[0].text
        assert "@article" in r3[0].text

    @patch("semtero.search_engine.Client")
    @patch("semtero.search_engine.VectorStore")
    def test_search_best_sentences_emits_progress_updates(
        self, mock_vector_store_cls, mock_client_cls
    ):
        mock_client = MagicMock()
        mock_client.embeddings.return_value = {"embedding": [0.0, 0.0, 1.0]}
        mock_client_cls.return_value = mock_client

        config = Config()
        config.EMBEDDING_DIMENSIONS = 0
        engine = SearchEngine(config)

        mock_vs = MagicMock()
        mock_vs.get_detected_dimension.return_value = None
        mock_vs.search_sentence_ids.return_value = (
            ["s1"],
            [0.8],
            [{"document_key": "docA"}],
        )
        mock_vs.get_sentence_texts_by_ids.return_value = {"s1": "hello"}
        engine.vector_store = mock_vs

        updates = []
        results = engine.search_best_sentences(
            "q",
            top_sentences=1,
            progress_callback=updates.append,
        )

        assert len(results) == 1
        assert updates[0]["message"] == "Embedding sentence"
        assert updates[-1]["message"] == "Found 1 similar sentence"
        assert updates[-1]["similar_sentences"] == 1


class TestGetStats:
    """Test suite for get_stats method."""

    def test_get_stats(self):
        config = Config()

        mock_vs = MagicMock()
        mock_vs.get_sentence_count.return_value = 50
        mock_vs.get_embedded_documents.return_value = {
            "doc1": 10,
            "doc2": 20,
            "doc3": 30,
        }

        engine = SearchEngine(config)
        engine.vector_store = mock_vs

        result = engine.get_stats()

        assert result["total_sentences"] == 50
        assert result["embedded_documents"] == 3


class TestDimensionMismatch:
    """Test suite for dimension mismatch detection."""

    def test_dimension_mismatch_warning(self):
        config = Config()
        config.EMBEDDING_DIMENSIONS = 1024

        with patch("semtero.search_engine.VectorStore") as mock_vs_cls:
            mock_vs = MagicMock()
            mock_vs.has_dimension_mismatch.return_value = True
            mock_vs.get_detected_dimension.return_value = 768
            mock_vs_cls.return_value = mock_vs

            with patch("semtero.search_engine.logger") as mock_logger:
                _ = SearchEngine(config)
                assert mock_logger.warning.called


class TestSearchResultModel:
    """Test suite for SearchResult model in search context."""

    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        result = SearchResult(
            text="Sample text",
            document_title="Doc Title",
            section_title="Section Title",
            zotero_key="ABC123",
            relevance_score=0.8,
            rerank_score=0.9,
        )

        assert result.text == "Sample text"
        assert result.relevance_score == 0.8

    def test_search_result_default_scores(self):
        """Test default score values."""
        result = SearchResult(
            text="Text", document_title="", section_title="", zotero_key=""
        )

        assert result.rerank_score is None or isinstance(result.rerank_score, float)
