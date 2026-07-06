"""Tests for citation_extractor module."""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from semtero.citation_extractor import (
    _bibtex_key_from_authors_year,
    _expand_numbers,
    _split_authors,
    extract_citation_metadata,
    extract_citation_numbers_from_sentence,
    extract_page_text_from_pymupdf4llm,
    find_references_start_page,
    parse_references_from_pages,
    reference_text_to_bibtex,
)


class TestExpandNumbers:
    def test_single_number(self):
        assert _expand_numbers("5") == [5]

    def test_multiple_numbers(self):
        assert _expand_numbers("5, 2, 35") == [2, 5, 35]

    def test_number_range(self):
        assert _expand_numbers("7-9") == [7, 8, 9]

    def test_range_with_multiple(self):
        assert _expand_numbers("7-9, 12") == [7, 8, 9, 12]

    def test_en_dash_range(self):
        assert _expand_numbers("3–7") == [3, 4, 5, 6, 7]

    def test_semicolons(self):
        assert _expand_numbers("1; 3; 5") == [1, 3, 5]

    def test_mixed_separators(self):
        assert _expand_numbers("1-3, 7; 9-10") == [1, 2, 3, 7, 9, 10]

    def test_range_too_large_returns_empty(self):
        assert _expand_numbers("1-3000") == []

    def test_boundary_range_returns_numbers(self):
        assert _expand_numbers("1-2000") == list(range(1, 2001))

    def test_empty_string(self):
        assert _expand_numbers("") == []

    def test_whitespace_padded(self):
        assert _expand_numbers("  5 , 7  ") == [5, 7]

    def test_em_dash_range(self):
        assert _expand_numbers("10–12") == [10, 11, 12]

    def test_zero(self):
        assert _expand_numbers("0") == [0]

    def test_reversed_range_is_empty(self):
        assert _expand_numbers("5-3") == []


class TestExtractCitationNumbers:
    def test_simple(self):
        s = "Most models have an encoder-decoder structure [5, 2, 35]."
        assert extract_citation_numbers_from_sentence(s) == [2, 5, 35]

    def test_ranges_and_invalid(self):
        s = "See [7-9, 12] and also [not a citation] and [3; 4]."
        assert extract_citation_numbers_from_sentence(s) == [3, 4, 7, 8, 9, 12]

    def test_mid_sentence_multiple_groups(self):
        s = "models such as [17, 18] and [9]."
        assert extract_citation_numbers_from_sentence(s) == [9, 17, 18]

    def test_no_citations(self):
        s = "This sentence has no citations."
        assert extract_citation_numbers_from_sentence(s) == []

    def test_empty_string(self):
        assert extract_citation_numbers_from_sentence("") == []

    def test_citation_with_spaces(self):
        s = "Previous work [ 5 , 12 ] is relevant."
        assert extract_citation_numbers_from_sentence(s) == [5, 12]

    def test_partially_invalid_bracket(self):
        s = "Mixed [5, a, 12] and [3]."
        assert extract_citation_numbers_from_sentence(s) == [3]

    def test_single_bracket(self):
        s = "See [5]."
        assert extract_citation_numbers_from_sentence(s) == [5]

    def test_consecutive_brackets(self):
        s = "Multiple [5][12] citations."
        assert extract_citation_numbers_from_sentence(s) == [5, 12]

    def test_three_digit_range(self):
        s = "Large citations [100-105]."
        assert extract_citation_numbers_from_sentence(s) == [100, 101, 102, 103, 104, 105]

    def test_none_input(self):
        assert extract_citation_numbers_from_sentence(None) == []  # noqa


class TestExtractPageTextFromPymupdf4llm:
    def _mock_pdf_path(self, monkeypatch, tmp_path):
        path = tmp_path / "test.pdf"
        path.write_text("fake pdf content")
        return str(path)

    @patch("semtero.citation_extractor.pymupdf4llm")
    def test_returns_parsed_pages(self, mock_pymupdf, tmp_path):
        pdf_path = self._mock_pdf_path(MagicMock(), tmp_path)
        mock_pymupdf.to_markdown.return_value = [
            {"page": 1, "text": "Page one content."},
            {"page": 2, "text": "Page two content."},
        ]

        result = extract_page_text_from_pymupdf4llm(pdf_path)
        assert len(result) == 2
        assert result[0]["page"] == 1
        assert result[1]["text"] == "Page two content."

    def test_nonexistent_file_returns_empty(self):
        result = extract_page_text_from_pymupdf4llm("/nonexistent/file.pdf")
        assert result == []

    @patch("semtero.citation_extractor.pymupdf4llm")
    def test_handles_exception_gracefully(self, mock_pymupdf, tmp_path):
        pdf_path = self._mock_pdf_path(MagicMock(), tmp_path)
        mock_pymupdf.to_markdown.side_effect = RuntimeError("PDF parse error")
        result = extract_page_text_from_pymupdf4llm(pdf_path)
        assert result == []

    @patch("semtero.citation_extractor.pymupdf4llm")
    def test_single_string_return(self, mock_pymupdf, tmp_path):
        pdf_path = self._mock_pdf_path(MagicMock(), tmp_path)
        mock_pymupdf.to_markdown.return_value = "Plain markdown text"
        result = extract_page_text_from_pymupdf4llm(pdf_path)
        assert len(result) == 1
        assert result[0]["page"] == 1
        assert result[0]["text"] == "Plain markdown text"

    @patch("semtero.citation_extractor.pymupdf4llm")
    def test_empty_list(self, mock_pymupdf, tmp_path):
        pdf_path = self._mock_pdf_path(MagicMock(), tmp_path)
        mock_pymupdf.to_markdown.return_value = []
        result = extract_page_text_from_pymupdf4llm(pdf_path)
        assert result == []


class TestFindReferencesStartPage:
    def test_references_heading_found(self):
        pages = [
            {"page": 1, "text": "Introduction."},
            {"page": 2, "text": "Method."},
            {"page": 3, "text": "References\n[1] A. Author. Title. 2020."},
            {"page": 4, "text": "Appendix."},
        ]
        assert find_references_start_page(pages) == 2

    def test_heading_case_insensitive(self):
        pages = [
            {"page": 1, "text": "Intro."},
            {"page": 2, "text": "references\n[1] A. Author. Title. 2020."},
        ]
        assert find_references_start_page(pages) == 1

    def test_fallback_to_last_page_heuristic(self):
        pages = [
            {"page": 1, "text": "Introduction."},
            {"page": 2, "text": "[1] A. Author. Title. 2020.\n[2] B. Author. Other. 2021."},
        ]
        pos = find_references_start_page(pages)
        assert pos == 1

    def test_no_references_returns_last_page(self):
        pages = [
            {"page": 1, "text": "Introduction."},
            {"page": 2, "text": "Conclusion."},
        ]
        assert find_references_start_page(pages) == 1

    def test_empty_pages(self):
        assert find_references_start_page([]) == 0

    def test_single_page(self):
        pages = [{"page": 1, "text": "References\n[1] Test."}]
        assert find_references_start_page(pages) == 0


class TestParseReferencesFromPages:
    def test_simple_references(self):
        pages = [{"page": 2, "text": "[1] First reference.\n[2] Second reference."}]
        refs = parse_references_from_pages(pages, 0)
        assert len(refs) == 2
        assert "First reference" in refs[1]
        assert "Second reference" in refs[2]

    def test_multiline_references(self):
        pages = [
            {"page": 2, "text": "[1] First reference.\n[2] Second reference that continues\n  on the next line."}
        ]
        refs = parse_references_from_pages(pages, 0)
        assert len(refs) == 2
        assert "on the next line" in refs[2]

    def test_hyphenated_word_continuation(self):
        pages = [{"page": 2, "text": "[1] This reference has a hyphen-\nated word."}]
        refs = parse_references_from_pages(pages, 0)
        assert "hyphenated" in refs[1]

    def test_skips_heading_lines(self):
        pages = [{"page": 2, "text": "References\n[1] A real reference."}]
        refs = parse_references_from_pages(pages, 0)
        assert 1 in refs

    def test_skips_standalone_page_numbers(self):
        pages = [{"page": 3, "text": "42\n[1] Real reference."}]
        refs = parse_references_from_pages(pages, 0)
        assert 1 in refs


class TestSplitAuthors:
    def test_simple_author(self):
        assert _split_authors("Smith") == ["Smith"]

    def test_last_name_first(self):
        result = _split_authors("Smith, John")
        assert "Smith" in result

    def test_comma_separated_names(self):
        result = _split_authors("Smith, John, Jones, Bob")
        assert len(result) >= 2

    def test_empty_string(self):
        assert _split_authors("") == []

    def test_strips_trailing_period(self):
        result = _split_authors("Smith, J.")
        assert result == ["Smith", "J"]

    def test_only_period(self):
        assert _split_authors(".") == []

    def test_single_name(self):
        assert _split_authors("Vaswani") == ["Vaswani"]

    def test_et_al_stripped(self):
        result = _split_authors("Vaswani, et al")
        assert "et al" not in result

    def test_three_names_with_comma_and_and(self):
        result = _split_authors("Vaswani, Jones, and Smith")
        assert len(result) == 2


class TestBibtexKeyFromAuthorsYear:
    def test_basic(self):
        assert _bibtex_key_from_authors_year(["Smith", "Jones"], "2020") == "smith2020"

    def test_no_year(self):
        assert _bibtex_key_from_authors_year(["Smith"], None) == "smith"

    def test_empty_authors(self):
        assert _bibtex_key_from_authors_year([], "2020") == "ref2020"

    def test_non_ascii_char_stripped(self):
        result = _bibtex_key_from_authors_year(["Müller"], "2021")
        assert "ü" not in result
        assert result.isascii()

    def test_last_name_first_format(self):
        assert (
            _bibtex_key_from_authors_year(["Smith, John", "Jones, Bob"], "2021")
            == "smith2021"
        )


class TestReferenceTextToBibtex:
    def test_minimal(self):
        ref = "Ashish Vaswani, Noam Shazeer, and Niki Parmar. Attention is all you need. In Advances in Neural Information Processing Systems, 2017."
        bib = reference_text_to_bibtex(ref, number=5)
        assert bib is not None
        assert bib.startswith("@")
        assert "title" in bib.lower()
        assert "author" in bib.lower()
        assert "2017" in bib

    def test_returns_none_for_empty(self):
        assert reference_text_to_bibtex("") is None

    def test_returns_none_for_whitespace(self):
        assert reference_text_to_bibtex("   ") is None

    def test_returns_none_for_single_part(self):
        assert reference_text_to_bibtex("Random text") is None

    def test_arxiv_ref(self):
        ref = "K. He et al. Deep residual learning. arXiv:1512.03385, 2015."
        bib = reference_text_to_bibtex(ref)
        assert bib is not None
        assert "@misc" in bib
        assert "eprint" in bib

    def test_doi_in_reference(self):
        ref = "A. Vaswani. Attention is all you need. NeurIPS, 2017. doi: 10.1234/abc123."
        bib = reference_text_to_bibtex(ref)
        assert bib is not None
        assert "doi" in bib


class TestExtractCitationMetadata:
    @patch("semtero.citation_extractor.extract_page_text_from_pymupdf4llm")
    def test_returns_dict_keyed_by_sentence(self, mock_extract, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("fake")
        # Body on page 1, references on page 2
        mock_extract.return_value = [
            {"page": 1, "text": "Previous work [1, 2] is relevant here."},
            {"page": 2, "text": "References\n[1] First reference.\n[2] Second reference."},
        ]

        result = extract_citation_metadata(str(pdf_path))
        assert len(result) > 0
        key = next(iter(result))
        assert result[key].citation_numbers

    @patch("semtero.citation_extractor.extract_page_text_from_pymupdf4llm")
    def test_empty_when_no_pages(self, mock_extract, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("fake")
        mock_extract.return_value = []
        assert extract_citation_metadata(str(pdf_path)) == {}

    @patch("semtero.citation_extractor.extract_page_text_from_pymupdf4llm")
    def test_skips_short_sentences(self, mock_extract, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("fake")
        mock_extract.return_value = [
            {"page": 1, "text": "Hi [1].\nReferences\n[1] A ref."}
        ]

        result = extract_citation_metadata(str(pdf_path))
        for key, meta in result.items():
            assert len(key.split()) > 3

    @patch("semtero.citation_extractor.extract_page_text_from_pymupdf4llm")
    def test_skips_text_in_references_section(self, mock_extract, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("fake")
        mock_extract.return_value = [
            {"page": 1, "text": "Important result [1].\nReferences\n[1] The reference text."}
        ]

        result = extract_citation_metadata(str(pdf_path))
        for key in result:
            assert "References" not in key
