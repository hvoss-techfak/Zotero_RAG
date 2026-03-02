"""Data models for ZoteroRAG."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


CitationReturnMode = Literal["sentence", "bibtex", "both"]


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
class Sentence:
    """Represents a single sentence chunk embedded for retrieval.

    This replaces the previous section/window hierarchy: the smallest retrieval unit
    is now a single sentence.

    Citation metadata:
        When extracted from scientific PDFs, a sentence may cite bibliography items
        like "... structure [5, 2, 35]". During embedding we keep:
        - citation_numbers: numeric IDs seen in bracket citations
        - referenced_bibtex: resolved BibTeX entries (best-effort)
        - referenced_texts: raw reference strings from the PDF bibliography (best-effort)

        These fields are optional and may be empty.
    """

    id: str  # composite: doc_key + "_sent_" + sentence_index
    document_id: str
    page: int
    page_section: int | None  # Position within page when using page splitting (1-N)
    sentence_index: int
    text: str
    is_embedded: bool = False

    citation_numbers: list[int] = field(default_factory=list)
    referenced_bibtex: list[str] = field(default_factory=list)
    referenced_texts: list[str] = field(default_factory=list)

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
    final_score: float = 0.0
    # Enhanced metadata fields
    bibtex: str = ""
    file_path: str = ""
    authors: list[str] = field(default_factory=list)
    date: str = ""
    item_type: str = ""

    cited_bibtex: list[str] = field(default_factory=list)
    cited_reference_texts: list[str] = field(default_factory=list)
    citation_numbers: list[int] = field(default_factory=list)

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
            "cited_bibtex": self.cited_bibtex,
            "cited_reference_texts": self.cited_reference_texts,
            "citation_numbers": self.citation_numbers,
        }


@dataclass
class EmbeddingStatus:
    """Represents the current embedding status."""

    total_documents: int = 0
    processed_documents: int = 0
    embedded_sections: int = 0
    embedded_sentences: int = 0
    pending_sections: int = 0
    is_running: bool = False

    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage (0-100)."""

        if self.total_documents == 0:
            return 0.0
        return (self.processed_documents / self.total_documents) * 100

    def __str__(self) -> str:
        return (
            f"Progress: {self.processed_documents}/{self.total_documents} "
            f"({self.progress_percentage:.1f}%) - Sections: {self.embedded_sections}, "
            f"Sentences: {self.embedded_sentences}"
        )
