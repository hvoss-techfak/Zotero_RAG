"""Extended tests for PDFProcessor - extraction and sectioning methods."""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path like main.py does
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from semtero.pdf_processor import PDFProcessor
from semtero.models import Sentence


class TestExtractMarkdown:
    """Test suite for extract_markdown method."""

    @pytest.fixture
    def processor(self):
        return PDFProcessor()

    def test_extract_markdown_nonexistent_file(self, processor):
        """Should return empty string for nonexistent file."""
        result = processor.extract_markdown("/nonexistent/file.pdf")
        assert result == ""

    @patch("semtero.pdf_processor.pymupdf4llm.to_markdown")
    def test_extract_markdown_with_layout(self, mock_to_markdown, processor):
        """Test extraction with layout preservation."""
        mock_to_markdown.return_value = "Mocked markdown content"

        with patch("pathlib.Path.exists", return_value=True):
            result = processor.extract_markdown("/test.pdf")

        assert result == "Mocked markdown content"
        mock_to_markdown.assert_called_once()

    @patch("semtero.pdf_processor.pymupdf4llm.to_markdown")
    def test_extract_markdown_without_layout(self, mock_to_markdown, processor):
        """Test extraction without layout preservation."""
        processor_no_layout = PDFProcessor(use_layout=False)

        with patch("pathlib.Path.exists", return_value=True):
            result = processor_no_layout.extract_markdown("/test.pdf")

        # Called without page_chunks
        call_args = mock_to_markdown.call_args[0][0]
        assert "/test.pdf" in call_args

    @patch("semtero.pdf_processor.pymupdf4llm.to_markdown")
    def test_extract_markdown_returns_list(self, mock_to_markdown, processor):
        """Test handling of list return from pymupdf4llm."""
        mock_to_markdown.return_value = [
            {"text": "Page 1 content"},
            {"text": "Page 2 content"},
        ]

        with patch("pathlib.Path.exists", return_value=True):
            result = processor.extract_markdown("/test.pdf")

        assert result == "Page 1 content\nPage 2 content"

    @patch("semtero.pdf_processor.pymupdf4llm.to_markdown")
    def test_extract_markdown_exception_handling(self, mock_to_markdown, processor):
        """Test exception handling during extraction."""
        mock_to_markdown.side_effect = Exception("PDF error")

        with patch("pathlib.Path.exists", return_value=True):
            result = processor.extract_markdown("/test.pdf")

        assert result == ""


class TestExtractQuarterSections:
    """Test suite for extract_quarter_sections method."""

    @pytest.fixture
    def processor(self):
        return PDFProcessor()

    def test_extract_quarter_sections_nonexistent_file(self, processor):
        """Should return empty list for nonexistent file."""
        result = processor.extract_sentences("/nonexistent/file.pdf")
        assert result == []

    @patch("semtero.pdf_processor.pymupdf4llm.to_markdown")
    def test_extract_quarter_sections_single_page(self, mock_to_markdown, processor):
        """Test single page PDF handling."""
        mock_to_markdown.return_value = "Single page text content."

        with patch("pathlib.Path.exists", return_value=True):
            result = processor.extract_sentences("/test.pdf")

        assert len(result) >= 1
        assert isinstance(result[0], Sentence)
        assert result[0].document_id == "test"

    @patch("semtero.pdf_processor.pymupdf4llm.to_markdown")
    def test_extract_quarter_sections_multiple_pages(self, mock_to_markdown, processor):
        """Test multiple page PDF handling."""
        mock_to_markdown.return_value = [
            {"page": 1, "text": "Page 1 line 1\nPage 1 line 2\nPage 1 line 3"},
            {"page": 2, "text": "Page 2 content here"},
        ]

        with patch("pathlib.Path.exists", return_value=True):
            result = processor.extract_sentences("/test.pdf")

        # With page_splits=4 and only ~3 lines on first page + some on second,
        # should create sections
        assert len(result) > 0

    @patch("semtero.pdf_processor.pymupdf4llm.to_markdown")
    def test_extract_quarter_sections_empty_pages(self, mock_to_markdown, processor):
        """Test handling of empty page content."""
        mock_to_markdown.return_value = [
            {"page": 1, "text": ""},
            {"page": 2, "text": ""},
        ]

        with patch("pathlib.Path.exists", return_value=True):
            result = processor.extract_sentences("/test.pdf")

        # Empty pages should produce no sections
        assert len(result) == 0

    @patch("semtero.pdf_processor.pymupdf4llm.to_markdown")
    def test_extract_quarter_sections_exception(self, mock_to_markdown, processor):
        """Test exception handling."""
        mock_to_markdown.side_effect = Exception("PDF error")

        with patch("pathlib.Path.exists", return_value=True):
            result = processor.extract_sentences("/test.pdf")

        assert result == []


class TestBlockquoteCleanup:
    """Test suite for _cleanup_blockquotes method."""

    @pytest.fixture
    def processor(self):
        return PDFProcessor()

    def test_cleanup_simple_blockquote(self, processor):
        text = "> Simple quote"
        result = processor._cleanup_blockquotes(text)
        assert ">" not in result

    def test_cleanup_nested_blockquote(self, processor):
        text = """> Level 1
>> Level 2
>>> Level 3"""

        result = processor._cleanup_blockquotes(text)

        # No > markers should remain
        for line in result.split("\n"):
            assert not line.startswith(">")

    def test_cleanup_multiple_quotes(self, processor):
        text = """> First quote
>
> Second quote"""

        result = processor._cleanup_blockquotes(text)

        assert "First quote" in result
        assert "Second quote" in result


class TestHorizontalRules:
    """Test suite for _remove_horizontal_rules method."""

    @pytest.fixture
    def processor(self):
        return PDFProcessor()

    def test_remove_dashes(self, processor):
        text = "Text\n---\nMore"
        result = processor._remove_horizontal_rules(text)

        assert "---" not in result

    def test_remove_stars(self, processor):
        text = "Text\n***\nMore"
        result = processor._remove_horizontal_rules(text)

        assert "***" not in result

    def test_remove_underscores(self, processor):
        text = "Text\n___\nMore"
        result = processor._remove_horizontal_rules(text)

        assert "___" not in result


class TestWhitespaceNormalization:
    """Test suite for _normalize_whitespace method."""

    @pytest.fixture
    def processor(self):
        return PDFProcessor()

    def test_normalize_multiple_newlines(self, processor):
        text = "Line 1\n\n\n\nLine 2"
        result = processor._normalize_whitespace(text)

        assert "\n\n\n" not in result

    def test_normalize_trailing_spaces(self, processor):
        text = "Text with trailing   \nMore text"
        result = processor._normalize_whitespace(text)

        for line in result.split("\n"):
            assert not line.endswith(" ")

    def test_preserve_single_newlines(self, processor):
        text = "Line 1\nLine 2"
        result = processor._normalize_whitespace(text)

        # Single newlines should be preserved
        assert "\n" in result


class TestProcessDocumentFunction:
    """Test suite for process_document function."""

    @patch("semtero.pdf_processor.PDFProcessor")
    def test_process_document_uses_default_processor(self, mock_processor_class):
        """Test that process_document creates PDFProcessor with defaults."""
        from semtero.pdf_processor import process_document
        from semtero.models import Document

        # Setup mock
        mock_processor = Mock()
        mock_processor.extract_quarter_sections.return_value = []
        mock_processor_class.return_value = mock_processor

        doc = Document(
            zotero_key="test123",
            title="Test Doc",
            authors=["Author"],
            pdf_path=Path("/test/doc.pdf"),
        )

        result = process_document(doc)

        # Should call extract_quarter_sections with string path
        mock_processor.extract_quarter_sections.assert_called_once()
