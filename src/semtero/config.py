"""Configuration management for ZoteroRAG."""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv


def _setup_env_file() -> None:
    """Ensure .env file exists by copying from .env.example if needed."""
    env_path = Path(".env")
    env_example_path = Path(".env.example")

    if not env_path.exists() and env_example_path.exists():
        shutil.copy(env_example_path, env_path)


# Setup .env file on module import
_setup_env_file()

# Load environment variables from .env file
load_dotenv()


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


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

        # Logging
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "WARNING")
        self.NOISY_LOG_LEVEL: str = os.getenv("NOISY_LOG_LEVEL", "WARNING")

        # Network / runtime binding
        self.APP_HOST: str = os.getenv("APP_HOST", "127.0.0.1")
        self.WEBUI_HOST: str = os.getenv("WEBUI_HOST", self.APP_HOST)

        # DOI + Zotero Connector import settings
        self.DOI_BIBTEX_BASE_URL: str = os.getenv(
            "DOI_BIBTEX_BASE_URL", "https://doi.org/"
        )
        self.DOI_BIBTEX_TIMEOUT_SECONDS: int = _get_int(
            "DOI_BIBTEX_TIMEOUT_SECONDS", 10
        )

        # Zotero Connector import endpoint is served by the local Zotero Connector.
        # Example: http://127.0.0.1:23119/connector/import
        self.ZOTERO_CONNECTOR_IMPORT_PATH: str = os.getenv(
            "ZOTERO_CONNECTOR_IMPORT_PATH", "/connector/import"
        )
        self.ZOTERO_CONNECTOR_TIMEOUT_SECONDS: int = _get_int(
            "ZOTERO_CONNECTOR_TIMEOUT_SECONDS", 15
        )

        # Optional default target collection key when importing items via the connector.
        # If empty, Zotero decides where to place the new item.
        self.ZOTERO_DEFAULT_IMPORT_COLLECTION_KEY: str = os.getenv(
            "ZOTERO_DEFAULT_IMPORT_COLLECTION_KEY", ""
        )

        # Ollama settings
        self.OLLAMA_BASE_URL: str = os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
        self.RERANKER_MODEL: str = os.getenv(
            "RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B"
        )
        self.RERANKER_GPU_MIN_VRAM_GB: float = max(
            0.0, _get_float("RERANKER_GPU_MIN_VRAM_GB", 8.0)
        )

        # Embedding dimensions: 0 means auto-detect from model
        self.EMBEDDING_DIMENSIONS: int = _get_int("EMBEDDING_DIMENSIONS", 2560)

        # Search parameters
        self.DEFAULT_TOP_X: int = _get_int("DEFAULT_TOP_X", 10)
        self.DEFAULT_TOP_Y: int = _get_int("DEFAULT_TOP_Y", 5)

        # Storage paths
        self.VECTOR_STORE_DIR = Path(
            os.getenv("VECTOR_STORE_DIR", str(self.__class__.VECTOR_STORE_DIR))
        )
        self.PDF_CACHE_PATH = Path(
            os.getenv("PDF_CACHE_PATH", str(self.__class__.PDF_CACHE_PATH))
        )

        # Embedding options
        self.AUTO_EMBED_SENTENCES: bool = (
            os.getenv("AUTO_EMBED_SENTENCES", "false").lower() == "true"
        )
        self.AUTO_REEMBED_INTERVAL_MINUTES: int = max(
            0, _get_int("AUTO_REEMBED_INTERVAL_MINUTES", 15)
        )
        self.EMBED_PROGRESS_INTERVAL_SEC: float = max(
            0.2, _get_float("EMBED_PROGRESS_INTERVAL_SEC", 1.0)
        )

        # Threading for embedding
        workers = _get_int("MAX_EMBEDDING_WORKERS", 9)
        if workers <= 0:
            workers = max(1, (os.cpu_count() or 2) // 2)
        self.MAX_EMBEDDING_WORKERS = workers

        # Batch embedding settings
        self.BATCH_EMBEDDING_SIZE: int = _get_int("BATCH_EMBEDDING_SIZE", 256)

        # Batch size for vector store write operations.
        self.BATCH_VECTOR_STORE_SIZE: int = _get_int("BATCH_VECTOR_STORE_SIZE", 5000)

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
        cls.VECTOR_STORE_DIR = cfg.VECTOR_STORE_DIR
        cls.PDF_CACHE_PATH = cfg.PDF_CACHE_PATH
        cls.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        cls.PDF_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    def get_zotero_headers(self) -> dict:
        """Get headers for Zotero API requests."""
        if self.ZOTERO_API_KEY:
            return {"Zotero-API-Key": self.ZOTERO_API_KEY}
        return {}


# Global config instance
config = Config()
