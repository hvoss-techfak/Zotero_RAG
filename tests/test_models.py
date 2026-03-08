"""Tests for data models."""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zoterorag.models import (
    Document,
    Sentence,
    SearchResult,
    EmbeddingStatus,
)


class TestDocument:
    """Tests for Document model."""
    
    def test_create_document_minimal(self):
        """Test creating a document with minimal required fields."""
        doc = Document(
            zotero_key="test_key",
            title="Test Title"
        )
        assert doc.zotero_key == "test_key"
        assert doc.title == "Test Title"
        assert doc.authors == []
        assert doc.pdf_path is None
        assert doc.added_date == ""
        assert doc.modified_date == ""
        assert doc.parent_item_key is None
        assert doc.group_id is None
    
    def test_create_document_full(self):
        """Test creating a document with all fields."""
        pdf_path = Path("/path/to/file.pdf")
        doc = Document(
            zotero_key="full_key",
            title="Full Title",
            authors=["Author One", "Author Two"],
            pdf_path=pdf_path,
            added_date="2024-01-01T00:00:00Z",
            modified_date="2024-01-02T00:00:00Z",
            parent_item_key="parent_123",
            group_id=42
        )
        assert doc.zotero_key == "full_key"
        assert doc.title == "Full Title"
        assert len(doc.authors) == 2
        assert doc.pdf_path == pdf_path
        assert doc.added_date == "2024-01-01T00:00:00Z"
        assert doc.modified_date == "2024-01-02T00:00:00Z"
        assert doc.parent_item_key == "parent_123"
        assert doc.group_id == 42
    
    def test_document_hash(self):
        """Test document hashing based on zotero_key."""
        doc1 = Document(zotero_key="key1", title="Title 1")
        doc2 = Document(zotero_key="key1", title="Different Title")
        doc3 = Document(zotero_key="key2", title="Title 1")
        
        assert hash(doc1) == hash(doc2)  # Same key, same hash
        assert hash(doc1) != hash(doc3)  # Different key, different hash
    
    def test_document_equality(self):
        """Dataclass equality is field-based; we only guarantee hash-by-key."""
        doc1 = Document(zotero_key="key1", title="Title 1")
        doc2 = Document(zotero_key="key1", title="Different Title")

        assert doc1 != doc2


class TestSentence:
    """Tests for Sentence model."""

    def test_create_sentence_minimal(self):
        sent = Sentence(
            id="doc_sent_0",
            document_id="doc",
            page=1,
            page_section=None,
            sentence_index=0,
            text="First sentence.",
        )
        assert sent.id == "doc_sent_0"
        assert sent.document_id == "doc"
        assert sent.page == 1
        assert sent.page_section is None
        assert sent.sentence_index == 0
        assert sent.text == "First sentence."
        assert sent.is_embedded is False

    def test_sentence_hash(self):
        s1 = Sentence(
            id="x",
            document_id="d1",
            page=1,
            page_section=1,
            sentence_index=1,
            text="a",
        )
        s2 = Sentence(
            id="x",
            document_id="d2",
            page=2,
            page_section=None,
            sentence_index=99,
            text="b",
        )
        assert hash(s1) == hash(s2)


class TestSearchResult:
    """Tests for SearchResult model."""
    
    def test_create_search_result_minimal(self):
        """Test creating a search result with minimal fields."""
        result = SearchResult(
            text="Found relevant text",
            document_title="Document Title",
            section_title="Section Title",
            zotero_key="key123"
        )
        assert result.text == "Found relevant text"
        assert result.document_title == "Document Title"
        assert result.section_title == "Section Title"
        assert result.zotero_key == "key123"
        assert result.relevance_score == 0.0
        assert result.rerank_score == 0.0
    
    def test_create_search_result_full(self):
        """Test creating a search result with all fields."""
        result = SearchResult(
            text="Full result text",
            document_title="Full Doc Title",
            section_title="Full Section",
            zotero_key="key456",
            relevance_score=0.85,
            rerank_score=0.92,
            bibtex="@article{test, title={Test}}",
            file_path="/path/to/file.pdf",
            authors=["Author One", "Author Two"],
            date="2024-01-15",
            item_type="journalArticle"
        )
        assert result.relevance_score == 0.85
        assert result.rerank_score == 0.92
        assert result.bibtex == "@article{test, title={Test}}"
        assert len(result.authors) == 2
    
    def test_search_result_to_dict(self):
        """Test conversion to dictionary."""
        result = SearchResult(
            text="text",
            document_title="doc",
            section_title="section",
            zotero_key="key",
            relevance_score=0.5,
            rerank_score=0.6
        )
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert d["text"] == "text"
        assert d["document_title"] == "doc"
        assert d["section_title"] == "section"
        assert d["zotero_key"] == "key"
        assert d["relevance_score"] == 0.5
        assert d["rerank_score"] == 0.6
    
    def test_search_result_to_dict_with_all_fields(self):
        """Test conversion to dictionary with all fields."""
        result = SearchResult(
            text="text",
            document_title="doc",
            section_title="section",
            zotero_key="key",
            relevance_score=0.5,
            rerank_score=0.6,
            bibtex="@article{test}",
            file_path="/path.pdf",
            authors=["A1", "A2"],
            date="2024-01-01",
            item_type="book"
        )
        d = result.to_dict()
        
        assert d["bibtex"] == "@article{test}"
        assert d["file_path"] == "/path.pdf"
        assert d["authors"] == ["A1", "A2"]
        assert d["date"] == "2024-01-01"
        assert d["item_type"] == "book"


class TestEmbeddingStatus:
    """Tests for EmbeddingStatus model."""
    
    def test_create_embedding_status_defaults(self):
        """Test default embedding status values."""
        status = EmbeddingStatus()
        
        assert status.total_documents == 0
        assert status.processed_documents == 0
        assert status.embedded_sections == 0
        assert status.embedded_sentences == 0
        assert status.pending_sections == 0
        assert status.is_running is False
        assert status.failed_documents == 0
        assert status.started_at == ""
        assert status.finished_at == ""
        assert status.last_error == ""

    def test_create_embedding_status_full(self):
        """Test creating embedding status with all fields."""
        status = EmbeddingStatus(
            total_documents=10,
            processed_documents=5,
            embedded_sections=100,
            embedded_sentences=500,
            pending_sections=20,
            is_running=True,
            failed_documents=1,
            started_at="2026-03-08T10:00:00+00:00",
            finished_at="",
            last_error="pdf missing",
        )
        
        assert status.total_documents == 10
        assert status.processed_documents == 5
        assert status.embedded_sections == 100
        assert status.embedded_sentences == 500
        assert status.pending_sections == 20
        assert status.is_running is True
        assert status.failed_documents == 1
        assert status.last_error == "pdf missing"

    def test_progress_percentage_zero_total(self):
        """Test progress percentage when total is zero."""
        status = EmbeddingStatus(total_documents=0, processed_documents=0)
        assert status.progress_percentage == 0.0
    
    def test_progress_percentage_partial(self):
        """Test progress percentage with partial completion."""
        status = EmbeddingStatus(total_documents=10, processed_documents=5)
        assert status.progress_percentage == 50.0
    
    def test_progress_percentage_full(self):
        """Test progress percentage when complete."""
        status = EmbeddingStatus(total_documents=10, processed_documents=10)
        assert status.progress_percentage == 100.0
    
    def test_embedding_status_str(self):
        """Test string representation of embedding status."""
        status = EmbeddingStatus(
            total_documents=10,
            processed_documents=5,
            embedded_sections=50,
            embedded_sentences=200,
            is_running=True
        )
        
        result = str(status)
        
        assert "Progress: 5/10" in result
        assert "50.0%" in result or "50%" in result
        assert "Sections: 50" in result
        assert "Sentences: 200" in result
    
    def test_embedding_status_str_not_running(self):
        """Test string representation when not running."""
        status = EmbeddingStatus(is_running=False)
        
        result = str(status)
        
        assert isinstance(result, str)

    def test_pending_documents_property(self):
        status = EmbeddingStatus(total_documents=8, processed_documents=3)
        assert status.pending_documents == 5

    def test_embedding_status_to_dict(self):
        status = EmbeddingStatus(total_documents=4, processed_documents=2, failed_documents=1)
        data = status.to_dict()

        assert data["pending_documents"] == 2
        assert data["progress_percentage"] == 50.0
        assert data["failed_documents"] == 1


class TestModelEdgeCases:
    """Test edge cases in models."""
    
    def test_document_pdf_path_optional(self):
        """Test that pdf_path can be None."""
        doc = Document(zotero_key="key", title="Title")
        assert doc.pdf_path is None
    
    def test_search_result_empty_authors_default(self):
        """Test that authors defaults to empty list."""
        result = SearchResult(
            text="text",
            document_title="doc",
            section_title="section",
            zotero_key="key"
        )
        assert result.authors == []
    
    def test_embedding_status_fractional_progress(self):
        """Test progress calculation with non-integer division."""
        status = EmbeddingStatus(total_documents=3, processed_documents=1)
        # 1/3 = 33.33...%
        assert abs(status.progress_percentage - 33.333333333) < 0.01

