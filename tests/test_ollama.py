"""Tests for Ollama integration points (mocked)."""

from unittest.mock import patch

import pytest

from zoterorag.config import Config


class TestOllamaConnection:
    """Test Ollama connectivity."""

    @patch("zoterorag.embedding_manager.ollama")
    def test_embed_batch_single_format(self, mock_ollama):
        """Test batch embedding returns one embedding per input."""
        from zoterorag.embedding_manager import EmbeddingManager

        mock_embedding1 = [0.1] * 384
        mock_embedding2 = [0.2] * 384

        mock_ollama.embeddings.side_effect = [
            {"embedding": mock_embedding1},
            {"embedding": mock_embedding2},
        ]

        config = Config()
        manager = EmbeddingManager(config)

        result = manager.embed_batch(["text1", "text2"])

        assert len(result) == 2
        assert result[0] == mock_embedding1
        assert result[1] == mock_embedding2

        # Ensure we never send list prompts (Pydantic expects prompt: str in some versions)
        for call in mock_ollama.embeddings.call_args_list:
            assert isinstance(call.kwargs.get("prompt"), str)

    @patch("zoterorag.embedding_manager.ollama")
    def test_embed_batch_with_chunking(self, mock_ollama):
        """Test batch embedding with chunking for large batches."""
        from zoterorag.embedding_manager import EmbeddingManager

        mock_embeddings = [[float(i)] * 384 for i in range(10)]
        mock_ollama.embeddings.side_effect = [{"embedding": e} for e in mock_embeddings]

        config = Config()
        manager = EmbeddingManager(config)

        texts = [f"text{i}" for i in range(10)]
        result = manager.embed_batch(texts)

        assert len(result) == 10
        assert result == mock_embeddings
        assert mock_ollama.embeddings.call_count == 10

    @patch("zoterorag.embedding_manager.ollama")
    def test_embed_batch_fallback_on_error(self, mock_ollama):
        """Test batch embedding continues on per-item error."""
        from zoterorag.embedding_manager import EmbeddingManager

        mock_embedding = [0.1] * 384
        mock_ollama.embeddings.side_effect = [
            Exception("Item failed"),
            {"embedding": mock_embedding},
        ]

        config = Config()
        manager = EmbeddingManager(config)

        result = manager.embed_batch(["text1", "text2"])

        assert len(result) == 2
        assert result[0] == []
        assert result[1] == mock_embedding

