"""Configuration management for ZoteroRAG."""

import os
from pathlib import Path


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class Config:
    """Application configuration loaded from environment variables."""

    # Class attributes exist for tests that patch these symbols.
    VECTOR_STORE_DIR: Path = Path("./data/vector_store")
    PDF_CACHE_PATH: Path = Path("./data/pdfs")

    def __init__(self):
        # Zotero settings
        self.ZOTERO_API_URL: str = os.getenv("ZOTERO_API_URL", "http://127.0.0.1:23119")
        self.ZOTERO_API_KEY: str = os.getenv("ZOTERO_API_KEY", "")
        self.ZOTERO_LIBRARY_ID: str | None = os.getenv("ZOTERO_LIBRARY_ID")

        # Ollama settings
        self.OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
        self.RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "dengcao/Qwen3-Reranker-0.6B:Q8_0")

        # Embedding dimensions: 0 means auto-detect from model
        self.EMBEDDING_DIMENSIONS: int = _get_int("EMBEDDING_DIMENSIONS", 512)

        # Search parameters
        self.DEFAULT_TOP_X: int = _get_int("DEFAULT_TOP_X", 10)
        self.DEFAULT_TOP_Y: int = _get_int("DEFAULT_TOP_Y", 5)

        # Storage paths
        self.VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", str(self.__class__.VECTOR_STORE_DIR)))
        self.PDF_CACHE_PATH = Path(os.getenv("PDF_CACHE_PATH", str(self.__class__.PDF_CACHE_PATH)))

        # Embedding options
        self.AUTO_EMBED_SENTENCES: bool = os.getenv("AUTO_EMBED_SENTENCES", "false").lower() == "true"

        # Threading for embedding
        workers = _get_int("MAX_EMBEDDING_WORKERS", 4)
        if workers <= 0:
            workers = max(1, (os.cpu_count() or 2) // 2)
        self.MAX_EMBEDDING_WORKERS = workers

        # Batch embedding settings
        self.BATCH_EMBEDDING_SIZE: int = _get_int("BATCH_EMBEDDING_SIZE", 128)

        # PDF processing options
        self.PAGE_SPLITS: int = max(1, _get_int("PAGE_SPLITS", 4))

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load config from file (for future use)."""
        return cls()

    @classmethod
    def ensure_dirs(cls) -> None:
        """Ensure required directories exist."""
        cfg = cls()
        cfg.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        cfg.PDF_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    def get_zotero_headers(self) -> dict:
        """Get headers for Zotero API requests."""
        if self.ZOTERO_API_KEY:
            return {"Zotero-API-Key": self.ZOTERO_API_KEY}
        return {}


# Global config instance
config = Config()
