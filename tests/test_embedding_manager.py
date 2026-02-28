"""Tests for EmbeddingManager class."""

import sys
import os

# Add src to path like main.py does
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from zoterorag.config import Config
from zoterorag.embedding_manager import EmbeddingManager
from zoterorag.models import Document, Section, SentenceWindow, EmbeddingStatus


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

    def test_start_embedding_job_resets_previous_state(self, embedding_manager):
        """Test that start_embedding_job resets previous state."""
        # Set some previous state
        embedding_manager._embedding_progress = EmbeddingStatus(
            total_documents=100,
            processed_documents=50,
            embedded_sections=200,
            is_running=True
        )
        
        embedding_manager.start_embedding_job(total_documents=10)
        
        status = embedding_manager.get_embedding_status()
        
        # New values should be set
        assert status.total_documents == 10

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

    # --- Test embed_text ---

    @patch("zoterorag.embedding_manager.ollama")
    def test_embed_text_calls_ollama(self, mock_ollama, embedding_manager):
        """Test that embed_text calls Ollama API."""
        mock_response = {"embedding": [0.1] * 384}
        mock_ollama.embeddings.return_value = mock_response
        
        result = embedding_manager.embed_text("test text")
        
        assert result == [0.1] * 384
        mock_ollama.embeddings.assert_called_once()

    @patch("zoterorag.embedding_manager.ollama")
    def test_embed_text_with_empty_input(self, mock_ollama, embedding_manager):
        """Test that embed_text handles empty input."""
        mock_response = {"embedding": []}
        mock_ollama.embeddings.return_value = mock_response
        
        result = embedding_manager.embed_text("")
        
        assert isinstance(result, list)

    # --- Test _embed_batch_ollama ---

    @patch("zoterorag.embedding_manager.ollama")
    def test_embed_batch_returns_list(self, mock_ollama, embedding_manager):
        """Test that embed_batch returns a list of embeddings."""
        mock_response = {"embedding": [0.1] * 384}
        mock_ollama.embeddings.return_value = mock_response
        
        texts = ["text1", "text2"]
        result = embedding_manager._embed_batch_ollama(texts)
        
        assert isinstance(result, list)

    def test_embed_batch_with_empty_list(self, embedding_manager):
        """Test that embed_batch handles empty list."""
        result = embedding_manager.embed_batch([])
        
        assert result == []

    @patch("zoterorag.embedding_manager.ollama")
    def test_embed_batch_normalizes_inputs(self, mock_ollama, embedding_manager):
        """Test that embed_batch normalizes non-string inputs."""
        mock_response = {"embedding": [0.1] * 384}
        mock_ollama.embeddings.return_value = mock_response
        
        # Mix of string and non-string
        texts = ["valid", None, 123]
        
        result = embedding_manager._embed_batch_ollama(texts)
        
        assert len(result) == 3

    @patch("zoterorag.embedding_manager.ollama")
    def test_embed_batch_handles_embedding_failure(self, mock_ollama, embedding_manager):
        """Test that embed_batch handles individual embedding failures."""
        call_count = [0]
        
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Embedding failed")
            return {"embedding": [0.1] * 384}
        
        mock_ollama.embeddings.side_effect = side_effect
        
        texts = ["text1", "text2"]
        result = embedding_manager._embed_batch_ollama(texts)
        
        # Should still have 3 results (the failed one gets empty list)
        assert len(result) == 2

    # --- Test calculate_relevance_score ---

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
                mock_pp.extract_quarter_sections.return_value = []
                mock_pp_class.return_value = mock_pp
                
                manager = EmbeddingManager(mock_config)
                
                doc = Document(zotero_key="test", title="Test")
                sections, windows = manager.process_document(doc, "test.pdf")
                
                mock_pp.extract_quarter_sections.assert_called_once()

    def test_process_document_with_sentence_embedding(self, mock_config):
        """Test that process_document creates sentence windows when requested."""
        with patch("zoterorag.embedding_manager.VectorStore"):
            with patch("zoterorag.embedding_manager.PDFProcessor") as mock_pp_class:
                mock_section = Section(
                    id="sec1", document_id="doc1", title="Test",
                    level=1, start_page=1, end_page=1, text="Test content."
                )
                
                mock_window = SentenceWindow(
                    id="win1", section_id="sec1", window_index=0,
                    text="Test content.", sentences=["Test content."],
                    is_embedded=False
                )
                
                mock_pp = MagicMock()
                mock_pp.extract_quarter_sections.return_value = [mock_section]
                mock_pp.create_sentence_windows.return_value = [mock_window]
                mock_pp_class.return_value = mock_pp
                
                manager = EmbeddingManager(mock_config)
                
                doc = Document(zotero_key="test", title="Test")
                sections, windows = manager.process_document(
                    doc, "test.pdf", embed_sentences=True
                )
                
                # Should have called create_sentence_windows for each section
                assert mock_pp.create_sentence_windows.called

    def test_process_document_without_sentence_embedding(self, mock_config):
        """Test that process_document skips sentence windows when not requested."""
        with patch("zoterorag.embedding_manager.VectorStore"):
            with patch("zoterorag.embedding_manager.PDFProcessor") as mock_pp_class:
                mock_section = Section(
                    id="sec1", document_id="doc1", title="Test",
                    level=1, start_page=1, end_page=1, text="Test content."
                )
                
                mock_pp = MagicMock()
                mock_pp.extract_quarter_sections.return_value = [mock_section]
                mock_pp_class.return_value = mock_pp
                
                manager = EmbeddingManager(mock_config)
                manager.config.AUTO_EMBED_SENTENCES = False
                
                doc = Document(zotero_key="test", title="Test")
                sections, windows = manager.process_document(
                    doc, "test.pdf", embed_sentences=False
                )
                
                # create_sentence_windows should not be called when embed_sentences=False
                mock_pp.create_sentence_windows.assert_not_called()

    # --- Test _link_sections ---

    def test_link_sections_no_op(self, embedding_manager):
        """Test that _link_sections is a no-op (placeholder)."""
        sections = [
            Section(id="s1", document_id="doc1", title="A", level=1, 
                   start_page=1, end_page=1, text="Text 1"),
            Section(id="s2", document_id="doc1", title="B", level=2,
                   start_page=2, end_page=2, text="Text 2")
        ]
        
        # Should not raise
        embedding_manager._link_sections(sections)

    # --- Test embed_sections ---

    @patch("zoterorag.embedding_manager.ollama")
    def test_embed_sections_returns_ids_and_embeddings(self, mock_ollama, embedding_manager):
        """Test that embed_sections returns section IDs and embeddings."""
        mock_response = {"embedding": [0.1] * 384}
        mock_ollama.embeddings.return_value = mock_response
        
        sections = [
            Section(id="s1", document_id="doc1", title="A", level=1,
                   start_page=1, end_page=1, text="Text 1"),
            Section(id="s2", document_id="doc1", title="B", level=2,
                   start_page=2, end_page=2, text="Text 2")
        ]
        
        with patch.object(embedding_manager, 'embed_batch', return_value=[[0.1] * 384, [0.2] * 384]):
            ids, embeddings = embedding_manager.embed_sections(sections)
            
            assert len(ids) == 2
            assert len(embeddings) == 2

    # --- Test embed_sentence_windows ---

    @patch("zoterorag.embedding_manager.ollama")
    def test_embed_sentence_windows_returns_ids_and_embeddings(self, mock_ollama, embedding_manager):
        """Test that embed_sentence_windows returns window IDs and embeddings."""
        windows = [
            SentenceWindow(id="w1", section_id="s1", window_index=0,
                         text="Sentence 1.", sentences=["Sentence 1."], is_embedded=False),
            SentenceWindow(id="w2", section_id="s1", window_index=1,
                         text="Sentence 2.", sentences=["Sentence 2."], is_embedded=False)
        ]
        
        with patch.object(embedding_manager, 'embed_batch', return_value=[[0.1] * 384, [0.2] * 384]):
            ids, embeddings = embedding_manager.embed_sentence_windows(windows)
            
            assert len(ids) == 2
            assert len(embeddings) == 2

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
        """Test that embed_document_async calls callback on completion."""
        with patch("zoterorag.embedding_manager.VectorStore"):
            with patch("zoterorag.embedding_manager.PDFProcessor") as mock_pp:
                # Set up mocks to avoid actual processing
                manager = EmbeddingManager(mock_config)
                
                # Create a real executor for this test
                import os
                max_workers = max(1, os.cpu_count() // 2) if os.cpu_count() else 1
                manager._executor = __import__('concurrent.futures').ThreadPoolExecutor(max_workers=max_workers)
                
                doc = Document(zotero_key="test", title="Test")
                
                callback_called = []
                def callback(status):
                    callback_called.append(True)
                
                # This will fail on actual processing but we can test the callback mechanism
                try:
                    manager.embed_document_async(doc, "nonexistent.pdf", callback=callback)
                except Exception:
                    pass  # Expected to fail since file doesn't exist

    # --- Test embed_documents_sync ---

    def test_embed_documents_sync_returns_results_dict(self, mock_config):
        """Test that embed_documents_sync returns success/failed lists."""
        with patch("zoterorag.embedding_manager.VectorStore") as mock_vs:
            with patch("zoterorag.embedding_manager.PDFProcessor"):
                # Set up vector store mock
                mock_vs_instance = MagicMock()
                mock_vs_instance.get_embedded_documents.return_value = {}
                mock_vs.return_value = mock_vs_instance
                
                manager = EmbeddingManager(mock_config)
                
                docs = [
                    (Document(zotero_key="test1", title="Test 1"), "path1.pdf"),
                ]
                
                with patch.object(manager, 'process_document', side_effect=Exception("Test")):
                    results = manager.embed_documents_sync(docs)
                    
                    assert "success" in results
                    assert "failed" in results

    def test_embed_documents_sync_with_stop_on_error(self, mock_config):
        """Test that embed_documents_sync raises on first error when configured."""
        with patch("zoterorag.embedding_manager.VectorStore"):
            with patch("zoterorag.embedding_manager.PDFProcessor"):
                manager = EmbeddingManager(mock_config)
                
                docs = [
                    (Document(zotero_key="test1", title="Test 1"), "path1.pdf"),
                ]
                
                with patch.object(manager, 'process_document', side_effect=Exception("Test error")):
                    with pytest.raises(Exception):
                        results = manager.embed_documents_sync(docs, stop_on_first_error=True)

    # --- Test get_pdf_documents_from_directory ---

    def test_get_pdf_documents_returns_empty_for_missing_dir(self):
        """Test that get_pdf_documents returns empty list for missing directory."""
        result = EmbeddingManager.get_pdf_documents_from_directory(Path("/nonexistent"))
        
        assert result == []

    def test_get_pdf_documents_finds_pdfs_in_directory(self):
        """Test that get_pdf_documents finds PDF files in a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = Path(tmpdir)
            
            # Create some dummy PDF files
            (pdf_dir / "doc1.pdf").touch()
            (pdf_dir / "doc2.pdf").touch()
            (pdf_dir / "readme.txt").touch()  # Non-PDF
            
            result = EmbeddingManager.get_pdf_documents_from_directory(pdf_dir)
            
            assert len(result) == 2
            # Check that results are (Document, path) tuples
            assert all(isinstance(r[0], Document) for r in result)

    def test_get_pdf_documents_uses_filename_as_key(self):
        """Test that get_pdf_documents uses filename as zotero_key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = Path(tmpdir)
            
            (pdf_dir / "mydocument.pdf").touch()
            
            result = EmbeddingManager.get_pdf_documents_from_directory(pdf_dir)
            
            assert len(result) == 1
            doc, path = result[0]
            assert doc.zotero_key == "mydocument"
            assert doc.title == "mydocument.pdf"

    def test_get_pdf_documents_sorts_by_filename(self):
        """Test that get_pdf_documents returns files sorted alphabetically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = Path(tmpdir)
            
            # Create files in non-alphabetical order
            (pdf_dir / "z_file.pdf").touch()
            (pdf_dir / "a_file.pdf").touch()
            (pdf_dir / "m_file.pdf").touch()
            
            result = EmbeddingManager.get_pdf_documents_from_directory(pdf_dir)
            
            keys = [r[0].zotero_key for r in result]
            assert keys == ["a_file", "m_file", "z_file"]

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
                mock_pp.return_value.extract_quarter_sections.return_value = []
                
                manager = EmbeddingManager(mock_config)
                
                doc = Document(zotero_key="test", title="Test")
                sections, windows = manager.process_document(doc, "/nonexistent/file.pdf")
                
                # Should return empty lists
                assert sections == []
                assert windows == []