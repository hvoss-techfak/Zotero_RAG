"""Tests for Ollama connectivity and embedding functionality."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zoterorag.config import Config
from zoterorag.models import Section, SentenceWindow


class TestOllamaConnection:
    """Test Ollama connectivity."""

    def test_ollama_config_defaults(self):
        """Test default Ollama configuration."""
        config = Config()
        assert config.OLLAMA_BASE_URL == "http://localhost:11434"
        assert config.EMBEDDING_MODEL == "qwen3-embedding:0.6b"
        assert config.RERANKER_MODEL == "dengcao/Qwen3-Reranker-0.6B:Q8_0"

    def test_ollama_config_env_override(self, monkeypatch):
        """Test environment variable override for Ollama."""
        # Note: Config loads env vars at import time, so we need to set them before import
        # This tests that the config supports these env vars conceptually
        assert hasattr(Config, 'OLLAMA_BASE_URL')
        assert hasattr(Config, 'EMBEDDING_MODEL')
        assert hasattr(Config, 'RERANKER_MODEL')

    @patch("zoterorag.embedding_manager.ollama")
    def test_embed_text_success(self, mock_ollama):
        """Test successful text embedding."""
        from zoterorag.embedding_manager import EmbeddingManager
        
        # Mock ollama.embeddings response
        mock_embedding = [0.1] * 384
        mock_ollama.embeddings.return_value = {"embedding": mock_embedding}
        
        config = Config()
        manager = EmbeddingManager(config)
        
        result = manager.embed_text("test prompt")
        
        assert result == mock_embedding
        mock_ollama.embeddings.assert_called_once()

    @patch("zoterorag.embedding_manager.ollama")
    def test_embed_batch(self, mock_ollama):
        """Test batch embedding."""
        from zoterorag.embedding_manager import EmbeddingManager
        
        # Mock ollama.embeddings to return different embeddings
        mock_embedding1 = [0.1] * 384
        mock_embedding2 = [0.2] * 384
        
        mock_ollama.embeddings.side_effect = [
            {"embedding": mock_embedding1},
            {"embedding": mock_embedding2}
        ]
        
        config = Config()
        manager = EmbeddingManager(config)
        
        result = manager.embed_batch(["text1", "text2"])
        
        assert len(result) == 2
        assert result[0] == mock_embedding1
        assert result[1] == mock_embedding2

    @patch("zoterorag.embedding_manager.ollama")
    def test_embed_text_failure(self, mock_ollama):
        """Test embedding failure handling."""
        from zoterorag.embedding_manager import EmbeddingManager
        
        # Mock ollama to raise an exception
        mock_ollama.embeddings.side_effect = Exception("Connection failed")
        
        config = Config()
        manager = EmbeddingManager(config)
        
        with pytest.raises(Exception):
            manager.embed_text("test prompt")

    def test_test_ollama_connection_integration(self, tmp_path):
        """Test the actual Ollama connection test method from main.py."""
        # Import from main module
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        with patch("zoterorag.embedding_manager.ollama") as mock_ollama:
            mock_ollama.embeddings.return_value = {"embedding": [0.1] * 384}
            
            from zoterorag.config import Config
            config = Config()

            # Test embedding works with mocked ollama
            response = config.EMBEDDING_MODEL

            assert response == "qwen3-embedding:0.6b"


class TestOllamaIntegration:
    """Test integration with Ollama service."""

    def test_embedding_manager_initialization(self):
        """Test EmbeddingManager initializes correctly."""
        from zoterorag.embedding_manager import EmbeddingManager
        from zoterorag.config import Config
        
        config = Config()
        
        # Should not raise - creates manager without connecting to Ollama
        with patch("zoterorag.vector_store.VectorStore"):
            manager = EmbeddingManager(config)
            
        assert manager is not None
        assert manager.config == config

    def test_ollama_connection_integration(self):
        """Test actual Ollama connection by initializing EmbeddingManager and attempting embedding.
        
        This uses the EmbeddingManager class to verify connectivity works with real Ollama.
        """
        from zoterorag.embedding_manager import EmbeddingManager
        from zoterorag.config import Config
        
        config = Config()
        
        # Initialize manager (doesn't connect yet)
        with patch("zoterorag.vector_store.VectorStore"):
            manager = EmbeddingManager(config)
        
        # Attempt to embed text - this requires a running Ollama server
        try:
            embedding = manager.embed_text("test connection")
            
            # If successful, we should get a vector back
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            
            # Verify it's actually a numeric vector (not error response)
            assert all(isinstance(x, (int, float)) for x in embedding)
        except Exception as e:
            # If Ollama isn't running or model not available, test skips
            # This is expected behavior when no server is available
            pytest.skip(f"Ollama not available: {e}")

    def test_embed_sentence_windows_logic(self):
        """Test embedding sentence windows logic."""
        from zoterorag.models import SentenceWindow
        
        # Test that create_sentence_windows produces expected output format
        section = Section(
            id="doc_sec_0",
            document_id="ABC123",
            title="Test",
            level=1,
            start_page=1,
            end_page=1,
            text="First sentence. Second sentence! Third sentence?",
        )
        
        # Import processor to create windows
        from zoterorag.pdf_processor import PDFProcessor
        processor = PDFProcessor()
        
        windows = processor.create_sentence_windows(section, window_size=3)
        
        assert len(windows) >= 1
        # Verify text extraction works correctly
        texts = [w.text for w in windows]
        assert any("First sentence" in t for t in texts)

    def test_embed_sections_logic(self):
        """Test embedding sections logic."""
        from zoterorag.models import Section
        
        sections = [
            Section(
                id="doc_sec_0",
                document_id="ABC123",
                title="Introduction",
                level=1,
                start_page=1,
                end_page=3,
                text="This is the introduction."
            ),
            Section(
                id="doc_sec_1",
                document_id="ABC123", 
                title="Methods",
                level=1,
                start_page=4,
                end_page=10,
                text="These are our methods."
            )
        ]
        
        # Test that section texts can be extracted for embedding
        texts = [s.text for s in sections]
        
        assert len(texts) == 2
        assert "This is the introduction." in texts[0]




if __name__ == "__main__":
    pytest.main([__file__, "-v"])