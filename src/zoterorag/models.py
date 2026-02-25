"""Data models for ZoteroRAG."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """Represents a document from Zotero."""
    zotero_key: str
    title: str
    authors: list[str] = field(default_factory=list)
    pdf_path: Path | None = None
    added_date: str = ""
    modified_date: str = ""
    parent_item_key: str | None = None  # For standalone attachments, track the parent paper
    group_id: int | None = None  # Which group library (None = user library)

    def __hash__(self) -> int:
        return hash(self.zotero_key)


@dataclass
class Section:
    """Represents a section within a document."""
    id: str  # composite: doc_key + "_" + section_index
    document_id: str
    title: str
    level: int  # heading depth (1 = h1, 2 = h2, etc.)
    start_page: int
    end_page: int
    text: str
    is_embedded: bool = False
    page_section: int | None = None  # Position within page when using line-count splitting (1-N)

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class SentenceWindow:
    """Represents a sliding window of 3 consecutive sentences within a section."""
    id: str  # composite: section_id + "_" + window_index
    section_id: str
    window_index: int
    sentences: list[str] = field(default_factory=list)  # 3 consecutive sentences
    text: str = ""  # combined text of all sentences
    is_embedded: bool = False

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class SearchResult:
    """Represents a search result with source attribution."""
    text: str
    document_title: str
    section_title: str
    zotero_key: str
    relevance_score: float = 0.0
    rerank_score: float = 0.0
    # Enhanced metadata fields
    bibtex: str = ""
    file_path: str = ""
    authors: list[str] = field(default_factory=list)
    date: str = ""
    item_type: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "document_title": self.document_title,
            "section_title": self.section_title,
            "zotero_key": self.zotero_key,
            "relevance_score": self.relevance_score,
            "rerank_score": self.rerank_score,
            "bibtex": self.bibtex,
            "file_path": self.file_path,
            "authors": self.authors,
            "date": self.date,
            "item_type": self.item_type,
        }


@dataclass
class EmbeddingStatus:
    """Represents the current embedding status."""
    total_documents: int = 0
    embedded_sections: int = 0
    embedded_sentences: int = 0
    pending_sections: int = 0
    is_running: bool = False