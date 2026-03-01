"""PDF processor using pymupdf4llm for text extraction and sentence chunking."""

import contextlib
import io
import logging
import re
from pathlib import Path
from typing import List

import pymupdf4llm

from .config import config
from .models import Document, Sentence
from .citation_extractor import extract_citation_metadata
from .citation_extractor import extract_citation_numbers_from_sentence

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Process PDFs to extract sentence chunks."""

    # Regex patterns for markdown headings
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(self, use_layout: bool = True, page_splits: int | None = None):
        """Initialize PDF processor.

        Args:
            use_layout: If True, use layout-preserving extraction for better
                       page structure analysis (headings, tables, etc.)
            page_splits: Number of parts to split each page into before chunking into sentences
                         (default from config)
        """
        self.use_layout = use_layout
        self.page_splits = page_splits if page_splits is not None else config.PAGE_SPLITS

    def extract_markdown(self, pdf_path: str) -> str:
        """Extract markdown text from PDF using pymupdf4llm."""
        path = Path(pdf_path)
        if not path.exists():
            return ""

        try:
            buf_out = io.StringIO()
            buf_err = io.StringIO()
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                if self.use_layout:
                    md_text = pymupdf4llm.to_markdown(
                        str(path),
                        page_chunks=True,
                        write_images=False,
                        extract_images=False,
                    )
                else:
                    md_text = pymupdf4llm.to_markdown(str(path))

            if isinstance(md_text, list):
                return "\n".join(chunk.get("text", "") for chunk in md_text)
            return str(md_text) if md_text else ""
        except Exception as e:
            logger.error("Error extracting from %s: %s", pdf_path, e)
            return ""

    # ---------- Markdown Sanitization Methods ----------

    def sanitize_markdown(self, text: str) -> str:
        """Convert markdown-formatted text to plain text."""
        if not text:
            return ""

        text = self._strip_heading_markers(text)
        text = self._remove_images(text)
        text = self._convert_links_to_text(text)
        text = self._extract_code_from_fenced_blocks(text)
        text = self._strip_code_formatting(text)
        text = self._strip_bold_italic(text)
        text = self._cleanup_blockquotes(text)
        text = self._remove_horizontal_rules(text)
        text = self._cleanup_lists(text)
        text = self._normalize_whitespace(text)

        return text.strip()

    def _strip_heading_markers(self, text: str) -> str:
        lines = []
        for line in text.split("\n"):
            stripped = re.sub(r"^#{1,6}\s+", "", line)
            lines.append(stripped)
        return "\n".join(lines)

    def _extract_code_from_fenced_blocks(self, text: str) -> str:
        result = []
        in_code_block = False
        lines = text.split("\n")

        for line in lines:
            if re.match(r"^```", line):
                in_code_block = not in_code_block
                continue

            result.append(line)

        return "\n".join(result)

    def _remove_images(self, text: str) -> str:
        return re.sub(r"!\[([^\]]*)\]\([^)]+\)", "", text)

    def _convert_links_to_text(self, text: str) -> str:
        return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    def _strip_code_formatting(self, text: str) -> str:
        return re.sub(r"`([^`]+)`", r"\1", text)

    def _strip_bold_italic(self, text: str) -> str:
        text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"(?<!\w)\*([^\*]+)\*(?!\w)", r"\1", text)
        text = re.sub(r"(?<![a-zA-Z])_([^_]+)_(?![a-zA-Z])", r"\1", text)
        return text

    def _cleanup_blockquotes(self, text: str) -> str:
        lines = []
        for line in text.split("\n"):
            cleaned = re.sub(r"^>+\s*", "", line)
            lines.append(cleaned)
        return "\n".join(lines)

    def _remove_horizontal_rules(self, text: str) -> str:
        return re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    def _cleanup_lists(self, text: str) -> str:
        lines = []
        for line in text.split("\n"):
            cleaned = re.sub(r"^[\s]*([-*+]|\d+\.)\s+", "", line)
            lines.append(cleaned)
        return "\n".join(lines)

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" +", " ", text)
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines)

    def extract_quarter_sections(self, pdf_path: str | Path) -> List[Sentence]:
        """Backward-compatible alias for `extract_sentences` to avoid breaking existing code."""
        return self.extract_sentences(pdf_path)

    def extract_sentences(self, pdf_path: str | Path) -> List[Sentence]:
        """Extract per-page sentence chunks.

        This keeps the robust page-chunk extraction and page-splitting logic formerly
        used for "sections". The difference is that we now emit a Sentence per
        sentence (the embedding unit), rather than a larger Section/window.

        Returns:
            List[Sentence] objects.
        """
        path = Path(pdf_path)
        if not path.exists():
            return []

        # Best-effort citation extraction. This uses PyMuPDF directly and may produce
        # slightly different sentence strings compared to pymupdf4llm. We attach
        # metadata by exact sentence match (best-effort) or by a whitespace-normalized
        # match fallback.
        citations_by_sentence: dict[str, object] = {}
        citations_by_normalized: dict[str, object] = {}
        try:
            extracted = extract_citation_metadata(path)
            citations_by_sentence = extracted
            citations_by_normalized = {self._normalize_ws(k): v for k, v in extracted.items()}
        except Exception as e:
            logger.debug("Citation extraction failed for %s: %s", path, e)

        try:
            buf_out = io.StringIO()
            buf_err = io.StringIO()
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                md_text = pymupdf4llm.to_markdown(
                    str(path),
                    page_chunks=True,
                    write_images=False,
                    extract_images=False,
                )
        except Exception as e:
            logger.error("Error extracting from %s: %s", pdf_path, e)
            return []

        if not isinstance(md_text, list):
            text = str(md_text) if md_text else ""
            sanitized_text = self.sanitize_markdown(text)
            return self._sentence_from_text(
                document_id=path.stem,
                page=1,
                page_section=None,
                text=sanitized_text,
                citations_by_sentence=citations_by_sentence,
                citations_by_normalized=citations_by_normalized,
            )

        out_sentences: List[Sentence] = []
        sentence_index = 0

        for chunk in md_text:
            page = int(chunk.get("page", 1))
            text = chunk.get("text", "")

            lines = [l for l in text.split("\n") if l.strip()]
            if not lines:
                continue

            # Combine lines into completed sentences (same logic as before)
            sentence_lines: list[str] = []
            current_sentence = ""
            for line in lines:
                current_sentence += " " + line.strip()
                if re.search(r"[.!?]$", line.strip()):
                    sentence_lines.append(current_sentence.strip())
                    current_sentence = ""
            if current_sentence.strip():
                sentence_lines.append(current_sentence.strip())

            # only keep sentences with more than 3 words (filter out noise)
            sentence_lines = [s for s in sentence_lines if len(s.split()) > 3]
            if not sentence_lines:
                continue

            # Re-split into sentences (defensive)
            normalized: list[str] = []
            for line in sentence_lines:
                split_sentences = re.split(r"(?<=[.!?])\s+", line)
                normalized.extend([s.strip() for s in split_sentences if s.strip()])

            if not normalized:
                continue

            all_sentences = normalized
            all_sentences = [self.sanitize_markdown(s) for s in all_sentences if s.strip()]
            if not all_sentences:
                continue

            for sentence in all_sentences:
                part_sentence = self._sentence_from_text(
                    document_id=path.stem,
                    page=page,
                    page_section=0,
                    text=sentence,
                    start_sentence_index=sentence_index,
                    citations_by_sentence=citations_by_sentence,
                    citations_by_normalized=citations_by_normalized,
                )
                out_sentences.extend(part_sentence)
                sentence_index += 1

        return out_sentences

    def _normalize_ws(self, s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def _sentence_from_text(
        self,
        document_id: str,
        page: int,
        page_section: int | None,
        text: str,
        start_sentence_index: int = 0,
        citations_by_sentence: dict[str, object] | None = None,
        citations_by_normalized: dict[str, object] | None = None,
    ) -> List[Sentence]:
        sentence_list = re.split(r"(?<=[.!?])\s+", text)
        sentence_list = [s.strip() for s in sentence_list if s.strip()]

        out: List[Sentence] = []
        for i, sent in enumerate(sentence_list):
            idx = start_sentence_index + i

            citation_numbers: list[int] = []
            referenced_texts: list[str] = []
            referenced_bibtex: list[str] = []

            meta = None
            if citations_by_sentence:
                meta = citations_by_sentence.get(sent)
                if meta is None and citations_by_normalized:
                    meta = citations_by_normalized.get(self._normalize_ws(sent))

            if meta is not None:
                # citation_extractor.CitationMetadata dataclass
                citation_numbers = list(getattr(meta, "citation_numbers", []) or [])
                referenced_texts = list(getattr(meta, "referenced_texts", []) or [])
                referenced_bibtex = list(getattr(meta, "referenced_bibtex", []) or [])
            else:
                # Fallback: parse bracketed numeric citations directly from the sentence.
                # This handles cases where citation groups appear mid-sentence and/or
                # the sentence text doesn't exactly match what PyMuPDF extracted.
                try:
                    citation_numbers = extract_citation_numbers_from_sentence(sent)
                except Exception:
                    citation_numbers = []

            out.append(
                Sentence(
                    id=f"{document_id}_sent_{idx}",
                    document_id=document_id,
                    page=page,
                    page_section=page_section,
                    sentence_index=idx,
                    text=sent,
                    is_embedded=False,
                    citation_numbers=citation_numbers,
                    referenced_texts=referenced_texts,
                    referenced_bibtex=referenced_bibtex,
                )
            )

        return out


def process_document(document: Document) -> List[Sentence]:
    """Process a document to extract sentences."""
    processor = PDFProcessor()
    return processor.extract_quarter_sections(str(document.pdf_path))
