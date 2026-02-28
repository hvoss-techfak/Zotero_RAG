"""Tests for VectorStore class."""

import sys
import os

# Add src to path like main.py does
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from zoterorag.models import Sentence


class TestVectorStore:
    """Test suite for VectorStore class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_chroma_client(self):
        """Mock ChromaDB client."""
        with patch("zoterorag.vector_store.chromadb.PersistentClient") as mock:
            mock_client = MagicMock()
            mock.return_value = mock_client
            
            # Mock collections
            sentences_collection = MagicMock()
            
            mock_client.get_collection.return_value = sentences_collection
            mock_client.create_collection.side_effect = lambda name: (
                sentences_collection
            )
            
            # Setup count and get methods
            type(sentences_collection).count = PropertyMock(return_value=0)
            
            yield mock_client, sentences_collection

    @pytest.fixture
    def vector_store(self, temp_dir, mock_chroma_client):
        """Create a VectorStore instance with mocked ChromaDB."""
        with patch("zoterorag.vector_store.chromadb.PersistentClient"):
            from zoterorag.vector_store import VectorStore
            store = VectorStore(persist_directory=str(temp_dir))
            yield store

    # --- Test _detect_dimensions ---

    def test_detect_dimensions_with_existing_embeddings(self, temp_dir):
        """Test dimension detection when embeddings exist."""
        with patch("zoterorag.vector_store.chromadb.PersistentClient") as mock:
            mock_client = MagicMock()
            mock.return_value = mock_client

            sentences_collection = MagicMock()

            def get_collection_side_effect(name):
                return sentences_collection

            mock_client.get_collection.side_effect = get_collection_side_effect

            # Mock embeddings with dimension 384
            import numpy as np
            mock_embedding = np.array([0.1] * 384)

            sentences_collection.get.return_value = {
                "ids": ["sent_1"],
                "embeddings": [mock_embedding],
            }

            def get_or_create(name):
                return sentences_collection

            mock_client.get_or_create_collection.side_effect = get_or_create

            from zoterorag.vector_store import VectorStore

            store = VectorStore(persist_directory=str(temp_dir))

            assert store._detected_sentence_dim == 384

    def test_detect_dimensions_with_empty_collections(self, temp_dir):
        """Test dimension detection with empty collections."""
        with patch("zoterorag.vector_store.chromadb.PersistentClient") as mock:
            mock_client = MagicMock()
            mock.return_value = mock_client

            sentences_collection = MagicMock()
            mock_client.get_collection.return_value = sentences_collection

            # Empty embeddings
            sentences_collection.get.return_value = {"ids": [], "embeddings": []}

            from zoterorag.vector_store import VectorStore

            store = VectorStore(persist_directory=str(temp_dir))

            assert store._detected_sentence_dim is None

    # --- Test get_detected_dimension ---

    def test_get_detected_dimension_returns_stored_value(self, vector_store):
        """Test that get_detected_dimension returns the detected value."""
        vector_store._detected_sentence_dim = 768
        assert vector_store.get_detected_dimension() == 768
        
        vector_store._detected_sentence_dim = None
        assert vector_store.get_detected_dimension() is None

    # --- Test has_dimension_mismatch ---

    def test_has_dimension_mismatch_returns_true_when_different(self, vector_store):
        """Test dimension mismatch detection when dimensions differ."""
        vector_store._detected_sentence_dim = 384
        assert vector_store.has_dimension_mismatch(768) is True

    def test_has_dimension_mismatch_returns_false_when_same(self, vector_store):
        """Test dimension mismatch returns false when dimensions match."""
        vector_store._detected_sentence_dim = 768
        assert vector_store.has_dimension_mismatch(768) is False

    def test_has_dimension_mismatch_returns_false_when_no_existing_data(self, vector_store):
        """Test dimension mismatch returns false when no existing data."""
        vector_store._detected_sentence_dim = None
        assert vector_store.has_dimension_mismatch(768) is False

    # --- Test add_sentences ---

    def test_add_sentences_with_valid_data(self, temp_dir):
        """Test adding sentences with embeddings."""
        from zoterorag.vector_store import VectorStore

        mock_client = MagicMock()
        sentences_collection = MagicMock()

        def get_or_create(name):
            return sentences_collection

        mock_client.get_or_create_collection.side_effect = get_or_create
        type(sentences_collection).count = PropertyMock(return_value=0)

        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))

            sentences = [
                Sentence(
                    id="doc_sent_0",
                    document_id="doc1",
                    page=1,
                    page_section=1,
                    sentence_index=0,
                    text="This is a sentence.",
                )
            ]
            embeddings = [[0.1] * 384]

            store.add_sentences(sentences, embeddings, "doc1")
            sentences_collection.upsert.assert_called_once()

    # --- Test delete_document ---

    def test_delete_document_calls_delete_on_collection(self, temp_dir):
        """Test that delete_document removes all vectors for a document."""
        from zoterorag.vector_store import VectorStore

        mock_client = MagicMock()
        sentences_collection = MagicMock()

        def get_or_create(name):
            return sentences_collection

        mock_client.get_or_create_collection.side_effect = get_or_create
        type(sentences_collection).count = PropertyMock(return_value=0)

        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            store.delete_document("doc1")
            sentences_collection.delete.assert_called_once_with(where={"document_key": "doc1"})

    # --- Test get_document_title ---

    def test_get_document_title_returns_title_when_found(self, temp_dir):
        """Titles are no longer stored in the vector DB."""
        from zoterorag.vector_store import VectorStore

        with patch("zoterorag.vector_store.chromadb.PersistentClient"):
            store = VectorStore(persist_directory=str(temp_dir))
            assert store.get_document_title("doc1") is None

    def test_get_document_title_returns_none_when_not_found(self, temp_dir):
        """Test that get_document_title returns None when document not found."""
        from zoterorag.vector_store import VectorStore
        
        mock_client = MagicMock()
        sentences_collection = MagicMock()

        # Empty result
        sentences_collection.get.return_value = {"metadatas": []}

        mock_client.get_collection.return_value = sentences_collection

        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            
            result = store.get_document_title("nonexistent")
            
            assert result is None

    # --- Test get_sentence_count ---

    def test_get_sentence_count_returns_count(self, temp_dir):
        """Test that get_sentence_count returns the number of sentences."""
        from zoterorag.vector_store import VectorStore

        mock_client = MagicMock()
        sentences_collection = MagicMock()
        type(sentences_collection).count = PropertyMock(return_value=100)

        def get_or_create(name):
            return sentences_collection

        mock_client.get_or_create_collection.side_effect = get_or_create

        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            assert store.get_sentence_count() == 100

    # --- Test persistence directory creation ---

    def test_persist_directory_created_on_init(self, temp_dir):
        """Test that persist directory is created on initialization."""
        new_dir = temp_dir / "new_vectordb"
        
        with patch("zoterorag.vector_store.chromadb.PersistentClient"):
            from zoterorag.vector_store import VectorStore
            store = VectorStore(persist_directory=str(new_dir))
            
            assert new_dir.exists()
            assert new_dir.is_dir()

    # --- Test thread safety ---

    def test_embedded_docs_lock_exists(self, temp_dir):
        """Test that the lock is created for thread safety."""
        with patch("zoterorag.vector_store.chromadb.PersistentClient"):
            from zoterorag.vector_store import VectorStore
            store = VectorStore(persist_directory=str(temp_dir))
            
            assert hasattr(store, "_embedded_docs_lock")

    # --- Test search_sentence_ids ---

    def test_search_sentence_ids_does_not_include_embeddings(self, temp_dir):
        """Query path should avoid returning embeddings for large-scale search."""
        from zoterorag.vector_store import VectorStore

        mock_client = MagicMock()
        sentences_collection = MagicMock()

        def get_or_create(name):
            return sentences_collection

        mock_client.get_or_create_collection.side_effect = get_or_create
        type(sentences_collection).count = PropertyMock(return_value=0)

        # dimension detection call
        sentences_collection.get.return_value = {"ids": [], "embeddings": []}

        sentences_collection.query.return_value = {
            "ids": [["s1", "s2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{"document_key": "d1"}, {"document_key": "d2"}]],
        }

        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            ids, distances, metas = store.search_sentence_ids([0.0, 1.0], top_k=2)

        assert ids == ["s1", "s2"]
        assert distances == [0.1, 0.2]
        assert metas[0]["document_key"] == "d1"

        # Ensure embeddings are not requested in query include
        _, kwargs = sentences_collection.query.call_args
        assert "include" in kwargs
        assert "embeddings" not in kwargs["include"]
        assert "distances" in kwargs["include"]

    def test_get_sentence_texts_by_ids(self, temp_dir):
        from zoterorag.vector_store import VectorStore

        mock_client = MagicMock()
        sentences_collection = MagicMock()
        mock_client.get_or_create_collection.side_effect = lambda name: sentences_collection
        type(sentences_collection).count = PropertyMock(return_value=0)

        # dimension detection call
        sentences_collection.get.return_value = {"ids": [], "embeddings": []}

        # get_sentence_texts_by_ids call
        sentences_collection.get.return_value = {
            "ids": ["a", "b"],
            "documents": ["A text", "B text"],
        }

        with patch("zoterorag.vector_store.chromadb.PersistentClient", return_value=mock_client):
            store = VectorStore(persist_directory=str(temp_dir))
            out = store.get_sentence_texts_by_ids(["a", "b"])

        assert out == {"a": "A text", "b": "B text"}
