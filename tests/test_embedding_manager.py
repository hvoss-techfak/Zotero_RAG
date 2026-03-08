"""Tests for EmbeddingManager class."""

import sys
import os

# Add src to path like main.py does
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

import pytest

from zoterorag.config import Config
from zoterorag.embedding_manager import EmbeddingManager
from zoterorag.models import Document, EmbeddingStatus


class TestEmbeddingManager:
    """Test suite for EmbeddingManager class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        config = MagicMock(spec=Config)
        config.VECTOR_STORE_DIR = "./data/vector_store"
        config.PAGE_SPLITS = 4
        config.EMBEDDING_MODEL = "qwen3-embedding-4b"
        config.RERANKER_MODEL = "Qwen3-Reranker-8B"
        config.EMBEDDING_DIMENSIONS = 0
        config.AUTO_EMBED_SENTENCES = False
        config.MAX_EMBEDDING_WORKERS = 2
        config.BATCH_EMBEDDING_SIZE = 32
        return config

    @pytest.fixture
    def embedding_manager(self, mock_config):
        """Create an EmbeddingManager with mocked dependencies."""
        with patch("zoterorag.embedding_manager.VectorStore"):
            with patch("zoterorag.embedding_manager.PDFProcessor"):
                manager = EmbeddingManager(mock_config)
                yield manager

    # --- Test __init__ ---

    def test_init_sets_config_and_dependencies(self, mock_config):
        """Test that initialization sets config and creates dependencies."""
        with patch("zoterorag.embedding_manager.VectorStore") as mock_vs:
            with patch("zoterorag.embedding_manager.PDFProcessor") as mock_pp:
                manager = EmbeddingManager(mock_config)
                
                assert manager.config == mock_config
                assert manager.vector_store is not None

    def test_init_with_zotero_client(self, mock_config):
        """Test that initialization accepts optional ZoteroClient."""
        zotero_client = MagicMock()
        
        with patch("zoterorag.embedding_manager.VectorStore"):
            with patch("zoterorag.embedding_manager.PDFProcessor"):
                manager = EmbeddingManager(mock_config, zotero_client=zotero_client)
                
                assert manager.zotero_client == zotero_client

    def test_init_creates_embedding_status(self, mock_config):
        """Test that initialization creates empty embedding status."""
        with patch("zoterorag.embedding_manager.VectorStore"):
            with patch("zoterorag.embedding_manager.PDFProcessor"):
                manager = EmbeddingManager(mock_config)
                
                assert isinstance(manager._embedding_progress, EmbeddingStatus)

    # --- Test get_embedding_status ---

    def test_get_embedding_status_returns_copy(self, embedding_manager):
        """Test that get_embedding_status returns a copy of status."""
        embedding_manager._embedding_progress.total_documents = 10
        embedding_manager._embedding_progress.processed_documents = 5
        
        status = embedding_manager.get_embedding_status()
        
        assert status.total_documents == 10
        assert status.processed_documents == 5

    def test_get_embedding_status_thread_safe(self, embedding_manager):
        """Test that get_embedding_status is thread-safe."""
        import threading
        
        # Set initial values
        embedding_manager._embedding_progress = EmbeddingStatus(
            total_documents=100,
            processed_documents=0,
            embedded_sections=0,
            embedded_sentences=0,
            pending_sections=0,
            is_running=False
        )
        
        results = []
        
        def read_status():
            for _ in range(100):
                status = embedding_manager.get_embedding_status()
                results.append(status.total_documents)
        
        threads = [threading.Thread(target=read_status) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All reads should see the same value
        assert all(r == 100 for r in results)

    # --- Test _update_progress ---

    def test_update_progress_updates_total_documents(self, embedding_manager):
        """Test that _update_progress updates total documents."""
        status = EmbeddingStatus(total_documents=10)
        
        embedding_manager._update_progress(status)
        
        assert embedding_manager._embedding_progress.total_documents == 10

    def test_update_progress_never_decreases_processed_count(self, embedding_manager):
        """Test that processed_documents never goes backwards."""
        # Set initial value
        embedding_manager._embedding_progress.processed_documents = 5
        
        # Try to set a lower value
        status = EmbeddingStatus(processed_documents=3)
        embedding_manager._update_progress(status)
        
        assert embedding_manager._embedding_progress.processed_documents == 5

    def test_update_progress_increments_sections(self, embedding_manager):
        """Test that embedded sections accumulate."""
        # Initial value
        embedding_manager._embedding_progress.embedded_sections = 0
        
        status1 = EmbeddingStatus(embedded_sections=10)
        embedding_manager._update_progress(status1)
        
        assert embedding_manager._embedding_progress.embedded_sections == 10
        
        status2 = EmbeddingStatus(embedded_sections=5)
        embedding_manager._update_progress(status2)
        
        # Should accumulate (10 + 5)
        assert embedding_manager._embedding_progress.embedded_sections == 15

    def test_update_progress_updates_is_running(self, embedding_manager):
        """Test that is_running flag can be updated."""
        embedding_manager._embedding_progress.is_running = True
        
        status1 = EmbeddingStatus(is_running=False)
        embedding_manager._update_progress(status1)
        
        assert embedding_manager._embedding_progress.is_running is False

    # --- Test start_embedding_job ---

    def test_start_embedding_job_initializes_status(self, embedding_manager):
        """Test that start_embedding_job initializes progress tracking."""
        embedding_manager.start_embedding_job(total_documents=25)
        
        status = embedding_manager.get_embedding_status()
        
        assert status.total_documents == 25
        assert status.processed_documents == 0
        assert status.is_running is True
        assert status.started_at

    def test_start_embedding_job_resets_previous_state(self, embedding_manager):
        """Test that start_embedding_job resets previous state."""
        # Set some previous state
        embedding_manager._embedding_progress = EmbeddingStatus(
            total_documents=100,
            processed_documents=50,
            embedded_sections=200,
            is_running=True,
            failed_documents=3,
            last_error="old error",
        )
        
        embedding_manager.start_embedding_job(total_documents=10)
        
        status = embedding_manager.get_embedding_status()
        
        # New values should be set
        assert status.total_documents == 10
        assert status.failed_documents == 0
        assert status.last_error == ""

    def test_mark_document_completed_updates_processed_and_failures(self, embedding_manager):
        embedding_manager.start_embedding_job(total_documents=2)

        snapshot = embedding_manager.mark_document_completed(
            embedded_sentences=12,
            failed=True,
            last_error="pdf missing",
        )

        assert snapshot.processed_documents == 1
        assert snapshot.embedded_sentences == 12
        assert snapshot.failed_documents == 1
        assert snapshot.last_error == "pdf missing"

    def test_finish_embedding_job_sets_finished_at(self, embedding_manager):
        embedding_manager.start_embedding_job(total_documents=1)
        status = embedding_manager.finish_embedding_job()

        assert status.is_running is False
        assert status.finished_at

    # --- Test executor property ---

    def test_executor_lazy_initialization(self, mock_config):
        """Test that executor is created lazily."""
        with patch("zoterorag.embedding_manager.VectorStore"):
            with patch("zoterorag.embedding_manager.PDFProcessor"):
                manager = EmbeddingManager(mock_config)
                
                assert manager._executor is None
                
                # Accessing executor should create it
                _ = manager.executor
                
                assert manager._executor is not None

    def test_executor_creates_with_max_workers(self, mock_config):
        """Test that executor uses max workers from config."""
        with patch("zoterorag.embedding_manager.VectorStore"):
            with patch("zoterorag.embedding_manager.PDFProcessor"):
                manager = EmbeddingManager(mock_config)
                
                _ = manager.executor
                
                assert manager._executor is not None

    # --- Test shutdown ---

    def test_shutdown_clears_executor(self, embedding_manager):
        """Test that shutdown clears the executor."""
        # Create executor first
        _ = embedding_manager.executor
        
        embedding_manager.shutdown()
        
        assert embedding_manager._executor is None

    # --- Test _get_embedding_options ---

    def test_get_embedding_options_with_dimensions(self, mock_config):
        """Test that options include dimensions when set in config."""
        mock_config.EMBEDDING_DIMENSIONS = 768
        
        with patch("zoterorag.embedding_manager.VectorStore"):
            with patch("zoterorag.embedding_manager.PDFProcessor"):
                manager = EmbeddingManager(mock_config)
                
                opts = manager._get_embedding_options()
                
                assert "dimensions" in opts
                assert opts["dimensions"] == 768

    def test_get_embedding_options_without_dimensions(self, mock_config):
        """Test that options don't include dimensions when not set."""
        mock_config.EMBEDDING_DIMENSIONS = 0
        
        with patch("zoterorag.embedding_manager.VectorStore"):
            with patch("zoterorag.embedding_manager.PDFProcessor"):
                manager = EmbeddingManager(mock_config)
                
                opts = manager._get_embedding_options()
                
                assert "dimensions" not in opts

    # --- Test _embed_batch_ollama ---

    def test_embed_batch_with_empty_list(self, embedding_manager):
        """Test that embed_batch handles empty list."""
        result = embedding_manager.embed_batch([])
        
        assert result == []


    def test_calculate_relevance_score_returns_zero_for_empty(self, embedding_manager):
        """Test that calculate_relevance_score returns 0 for empty input."""
        result = embedding_manager.calculate_relevance_score([])
        
        assert result == 0.0

    def test_calculate_relevance_score_all_positive(self, embedding_manager):
        """Test score calculation with all positive values."""
        embedding = [0.5] * 10
        
        result = embedding_manager.calculate_relevance_score(embedding)
        
        # Should be > 0 since all values are positive
        assert result > 0

    def test_calculate_relevance_score_all_negative(self, embedding_manager):
        """Test score calculation with all negative values."""
        embedding = [-0.5] * 10
        
        result = embedding_manager.calculate_relevance_score(embedding)
        
        # Should be < 0.5 since values are negative
        assert result < 0.5

    def test_calculate_relevance_score_mixed(self, embedding_manager):
        """Test score calculation with mixed positive/negative values."""
        embedding = [0.8, -0.3, 0.5, -0.2, 0.1]
        
        result = embedding_manager.calculate_relevance_score(embedding)
        
        # Should be between 0 and 1
        assert 0 <= result <= 1

    def test_calculate_relevance_score_zero_magnitude(self, embedding_manager):
        """Test score calculation when all values are zero."""
        embedding = [0.0] * 10
        
        result = embedding_manager.calculate_relevance_score(embedding)
        
        assert result == 0.0


    # --- Test process_document ---

    def test_process_document_calls_pdf_processor(self, mock_config):
        """Test that process_document uses PDFProcessor."""
        with patch("zoterorag.embedding_manager.VectorStore"):
            with patch("zoterorag.embedding_manager.PDFProcessor") as mock_pp_class:
                mock_pp = MagicMock()
                mock_pp.extract_sentences.return_value = []
                mock_pp_class.return_value = mock_pp

                manager = EmbeddingManager(mock_config)

                doc = Document(zotero_key="test", title="Test")
                sentences = manager.process_document(doc, "test.pdf")

                mock_pp.extract_sentences.assert_called_once_with("test.pdf", document_id="test")
                assert sentences == []

    # --- Test embed_document_async ---

    def test_embed_document_async_returns_future(self, mock_config):
        """Test that embed_document_async returns a Future."""
        with patch("zoterorag.embedding_manager.VectorStore"):
            with patch("zoterorag.embedding_manager.PDFProcessor"):
                manager = EmbeddingManager(mock_config)
                
                # Force executor creation
                _ = manager.executor
                
                doc = Document(zotero_key="test", title="Test")
                
                from concurrent.futures import Future
                
                result = manager.embed_document_async(doc, "test.pdf")
                
                assert isinstance(result, Future)

    def test_embed_document_async_with_callback(self, mock_config):
        """Test that embed_document_async returns a Future and accepts a callback."""
        with patch("zoterorag.embedding_manager.VectorStore"):
            with patch("zoterorag.embedding_manager.PDFProcessor"):
                manager = EmbeddingManager(mock_config)

                manager._executor = ThreadPoolExecutor(max_workers=1)

                doc = Document(zotero_key="test", title="Test")

                callback_called = []

                def callback(_status):
                    callback_called.append(True)

                fut = manager.embed_document_async(doc, "nonexistent.pdf", callback=callback)
                assert fut is not None

    # --- Test edge cases ---

    def test_embed_batch_with_none_in_list(self, embedding_manager):
        """Test that embed_batch handles None values in the list."""
        with patch.object(embedding_manager, '_embed_batch_ollama') as mock:
            mock.return_value = [[], [0.1] * 384]
            
            result = embedding_manager.embed_batch(["text", None])
            
            assert len(result) == 2

    def test_process_document_with_nonexistent_file(self, mock_config):
        """Test that process_document handles nonexistent files gracefully."""
        with patch("zoterorag.embedding_manager.VectorStore"):
            with patch("zoterorag.embedding_manager.PDFProcessor") as mock_pp:
                mock_pp.return_value.extract_sentences.return_value = []

                manager = EmbeddingManager(mock_config)

                doc = Document(zotero_key="test", title="Test")
                sentences = manager.process_document(doc, "/nonexistent/file.pdf")

                assert sentences == []
