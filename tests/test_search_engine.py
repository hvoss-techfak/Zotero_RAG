"""Tests for SearchEngine - sentence embedding search."""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, PropertyMock

# Add src to path like main.py does
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import ollama
from zoterorag.search_engine import SearchEngine
from zoterorag.config import Config
from zoterorag.models import SearchResult


class TestSearchEngineInitialization:
    """Test suite for SearchEngine initialization."""

    @patch('zoterorag.search_engine.VectorStore')
    def test_default_initialization(self, mock_vector_store):
        """Test default initialization."""
        config = Config()

        engine = SearchEngine(config)

        assert engine.config is not None
        assert engine.vector_store is not None


class TestQueryEmbedding:
    """Test suite for query embedding generation."""

    @patch('zoterorag.search_engine.ollama.embeddings')
    def test_get_query_embedding(self, mock_ollama_embeddings):
        """Test query embedding generation."""
        mock_response = {"embedding": [0.1, 0.2, 0.3]}
        mock_ollama_embeddings.return_value = mock_response

        config = Config()
        engine = SearchEngine(config)

        result = engine._get_query_embedding("test query")

        assert result == [0.1, 0.2, 0.3]
        mock_ollama_embeddings.assert_called_once()


class TestSearchBestSentences:
    """Search should use ANN ids+distances and only fetch top-k texts."""

    @patch('zoterorag.search_engine.ollama.embeddings')
    def test_search_best_sentences_uses_ids_and_fetches_texts(self, mock_ollama_embeddings):
        mock_ollama_embeddings.return_value = {"embedding": [0.0, 0.0, 1.0]}

        config = Config()
        engine = SearchEngine(config)

        mock_vs = MagicMock()
        mock_vs.search_sentence_ids.return_value = (
            ["s1", "s2"],
            [0.25, 0.5],
            [{"document_key": "docA"}, {"document_key": "docB"}],
        )
        mock_vs.get_sentence_texts_by_ids.return_value = {"s1": "hello", "s2": "world"}
        engine.vector_store = mock_vs

        results = engine.search_best_sentences("q", top_sentences=2)

        assert [r.text for r in results] == ["hello", "world"]
        mock_vs.search_sentence_ids.assert_called_once()
        mock_vs.get_sentence_texts_by_ids.assert_called_once_with(["s1", "s2"])
        assert all(isinstance(r, SearchResult) for r in results)


class TestGetStats:
    """Test suite for get_stats method."""

    def test_get_stats(self):
        config = Config()

        mock_vs = MagicMock()
        mock_vs.get_sentence_count.return_value = 50
        mock_vs.get_embedded_documents.return_value = {"doc1": 10, "doc2": 20, "doc3": 30}

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

        with patch('zoterorag.search_engine.VectorStore') as mock_vs_cls:
            mock_vs = MagicMock()
            mock_vs.has_dimension_mismatch.return_value = True
            mock_vs.get_detected_dimension.return_value = 768
            mock_vs_cls.return_value = mock_vs

            with patch('zoterorag.search_engine.logger') as mock_logger:
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
            rerank_score=0.9
        )

        assert result.text == "Sample text"
        assert result.relevance_score == 0.8

    def test_search_result_default_scores(self):
        """Test default score values."""
        result = SearchResult(
            text="Text",
            document_title="",
            section_title="",
            zotero_key=""
        )

        assert result.rerank_score is None or isinstance(result.rerank_score, float)
