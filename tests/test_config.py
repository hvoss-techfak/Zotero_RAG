"""Tests for Config class."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semtero.config import Config


class TestConfigFromEnvironment:
    """Test configuration loading from environment variables."""

    @patch.dict(
        os.environ,
        {
            "ZOTERO_API_URL": "https://api.zotero.org",
            "ZOTERO_API_KEY": "test_key_123",
            "ZOTERO_LIBRARY_ID": "12345",
            "OLLAMA_BASE_URL": "http://custom:11434",
            "EMBEDDING_MODEL": "custom-model:latest",
            "RERANKER_MODEL": "custom-reranker:latest",
            "RERANKER_GPU_MIN_VRAM_GB": "12.5",
            "RERANKER_BATCH_SIZE": "3",
            "EMBEDDING_DIMENSIONS": "1024",
            "DEFAULT_TOP_X": "20",
            "VECTOR_STORE_DIR": "/custom/vector_store",
            "PDF_CACHE_PATH": "/custom/pdfs",
            "AUTO_EMBED_SENTENCES": "true",
            "MAX_EMBEDDING_WORKERS": "8",
            "BATCH_EMBEDDING_SIZE": "256",
            "PAGE_SPLITS": "6",
        },
    )
    def test_env_overrides(self):
        """Test that environment variables override defaults."""
        config = Config()

        assert config.ZOTERO_API_URL == "https://api.zotero.org"
        assert config.ZOTERO_API_KEY == "test_key_123"
        assert config.ZOTERO_LIBRARY_ID == "12345"
        assert config.OLLAMA_BASE_URL == "http://custom:11434"
        assert config.EMBEDDING_MODEL == "custom-model:latest"
        assert config.RERANKER_GPU_MIN_VRAM_GB == 12.5
        assert config.RERANKER_BATCH_SIZE == 3
        assert config.EMBEDDING_DIMENSIONS == 1024
        assert config.DEFAULT_TOP_X == 20
        assert config.VECTOR_STORE_DIR == Path("/custom/vector_store")
        assert config.PDF_CACHE_PATH == Path("/custom/pdfs")
        assert config.AUTO_EMBED_SENTENCES is True
        assert config.MAX_EMBEDDING_WORKERS == 8
        assert config.BATCH_EMBEDDING_SIZE == 256
        assert config.PAGE_SPLITS == 6

    @patch.dict(os.environ, {"EMBEDDING_DIMENSIONS": "0"})
    def test_embedding_dimensions_zero_means_auto(self):
        """Test that EMBEDDING_DIMENSIONS=0 means auto-detect."""
        config = Config()
        assert config.EMBEDDING_DIMENSIONS == 0


class TestConfigMethods:
    """Test Config class methods."""

    @patch("semtero.config.Config.VECTOR_STORE_DIR", Path("/tmp/test_vector"))
    @patch("semtero.config.Config.PDF_CACHE_PATH", Path("/tmp/test_pdfs"))
    def test_ensure_dirs(self):
        """Test directory creation."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily override the paths to temp location
            Config.VECTOR_STORE_DIR = Path(tmpdir) / "vector"
            Config.PDF_CACHE_PATH = Path(tmpdir) / "pdfs"

            Config.ensure_dirs()

            assert Config.VECTOR_STORE_DIR.exists()
            assert Config.PDF_CACHE_PATH.exists()

    def test_get_zotero_headers_with_key(self):
        """Test Zotero headers when API key is set."""
        with patch.dict(os.environ, {"ZOTERO_API_KEY": "test_key"}):
            config = Config()
            headers = config.get_zotero_headers()
            assert headers == {"Zotero-API-Key": "test_key"}

    def test_get_zotero_headers_without_key(self):
        """Test Zotero headers when no API key is set."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear the env var
            if "ZOTERO_API_KEY" in os.environ:
                del os.environ["ZOTERO_API_KEY"]
            config = Config()
            headers = config.get_zotero_headers()
            assert headers == {}

    def test_from_file_returns_config(self):
        """Test that from_file returns a Config instance."""
        result = Config.from_file("/nonexistent/path")
        assert isinstance(result, Config)


class TestConfigEdgeCases:
    """Test edge cases in Config."""

    @patch.dict(os.environ, {"MAX_EMBEDDING_WORKERS": "0"})
    def test_max_workers_zero_uses_cpu_count(self):
        """Test that 0 workers defaults to CPU count // 2."""
        config = Config()
        # Should at least be 1
        assert config.MAX_EMBEDDING_WORKERS >= 1

    @patch.dict(os.environ, {"MAX_EMBEDDING_WORKERS": "abc"})
    def test_invalid_max_workers_falls_back_to_default(self):
        """Test that invalid worker count uses default."""
        # The int() conversion will use the default in the class definition
        # but here it's evaluated at import time, so this tests runtime behavior
        config = Config()
        # Should handle gracefully (use 1 as fallback)
        assert isinstance(config.MAX_EMBEDDING_WORKERS, int)

    @patch.dict(os.environ, {"RERANKER_GPU_MIN_VRAM_GB": "nope"})
    def test_invalid_reranker_vram_threshold_falls_back_to_default(self):
        """Test that invalid reranker VRAM thresholds use the safe default."""
        config = Config()
        assert config.RERANKER_GPU_MIN_VRAM_GB == 8.0

    @patch.dict(os.environ, {"RERANKER_BATCH_SIZE": "0"})
    def test_reranker_batch_size_is_clamped_to_at_least_one(self):
        """Test that reranker batch size never drops below one."""
        config = Config()
        assert config.RERANKER_BATCH_SIZE == 1

    def test_library_id_can_be_none(self):
        """Test that library ID can be None."""
        with patch.dict(os.environ, {}, clear=True):
            if "ZOTERO_LIBRARY_ID" in os.environ:
                del os.environ["ZOTERO_LIBRARY_ID"]
            config = Config()
            assert config.ZOTERO_LIBRARY_ID is None
