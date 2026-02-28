"""Configuration management for ZoteroRAG."""

import os
from pathlib import Path


class Config:
    """Application configuration loaded from environment variables."""

    # Zotero settings
    ZOTERO_API_URL: str = os.getenv("ZOTERO_API_URL", "http://127.0.0.1:23119")
    ZOTERO_API_KEY: str = os.getenv("ZOTERO_API_KEY", "")
    ZOTERO_LIBRARY_ID: str | None = os.getenv("ZOTERO_LIBRARY_ID")

    # Ollama settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "dengcao/Qwen3-Reranker-0.6B:Q8_0")
    
    # Embedding dimensions: 0 means auto-detect from Ollama model
    # Set explicitly if your model produces non-standard dimensions
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "128"))

    # Search parameters
    DEFAULT_TOP_X: int = 10  # Sections to retrieve in stage 1
    DEFAULT_TOP_Y: int = 5   # Sentence windows per section in stage 2

    # Storage paths  
    VECTOR_STORE_DIR: Path = Path(os.getenv("VECTOR_STORE_DIR", "./data/vector_store"))
    PDF_CACHE_PATH: Path = Path(os.getenv("PDF_CACHE_PATH", "./data/pdfs"))

    # Embedding options
    AUTO_EMBED_SENTENCES: bool = os.getenv("AUTO_EMBED_SENTENCES", "false").lower() == "true"

    # Threading for embedding
    MAX_EMBEDDING_WORKERS: int = int(os.getenv("MAX_EMBEDDING_WORKERS", 4))

    # Batch embedding settings
    BATCH_EMBEDDING_SIZE: int = int(os.getenv("BATCH_EMBEDDING_SIZE", "512"))  # Number of texts to embed in one batch call

    # PDF processing options
    PAGE_SPLITS: int = int(os.getenv("PAGE_SPLITS", "4"))  # Number of sections to split each page into

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load config from file (for future use)."""
        return cls()

    @classmethod
    def ensure_dirs(cls) -> None:
        """Ensure required directories exist."""
        cls.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        cls.PDF_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    def get_zotero_headers(self) -> dict:
        """Get headers for Zotero API requests."""
        if self.ZOTERO_API_KEY:
            return {"Zotero-API-Key": self.ZOTERO_API_KEY}
        # For local connector, use default headers
        return {}


# Global config instance
config = Config()
