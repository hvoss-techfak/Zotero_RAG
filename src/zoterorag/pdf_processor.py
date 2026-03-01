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

    def __init__(
        self,
        use_layout: bool = True,
        citation_fuzzy_threshold: float | None = None,
    ):
        """Initialize PDF processor.

        Args:
            use_layout: If True, use layout-preserving extraction for better
                       page structure analysis (headings, tables, etc.)
            page_splits: Number of parts to split each page into before chunking into sentences
                         (default from config)
            citation_fuzzy_threshold: Minimum similarity ratio (0.0-1.0) for fuzzy matching
                                      citations when exact/normalized match fails.
                                      Defaults to config value.
        """
        self.use_layout = use_layout
        self.meta_count = 0

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

    def extract_quarter_sections(
        self, pdf_path: str | Path, document_id: str | None = None
    ) -> List[Sentence]:
        """Backward-compatible alias for `extract_sentences` to avoid breaking existing code.
        
        Args:
            pdf_path: Path to the PDF file
            document_id: Optional override for the document ID used in sentence IDs.
                        If not provided, derives from filename (pdf_path stem).
        """
        return self.extract_sentences(pdf_path, document_id=document_id)

    def extract_sentences(
        self, pdf_path: str | Path, document_id: str | None = None
    ) -> List[Sentence]:
        """Extract per-page sentence chunks.

        This keeps the robust page-chunk extraction and page-splitting logic formerly
        used for "sections". The difference is that we now emit a Sentence per
        sentence (the embedding unit), rather than a larger Section/window.

        Args:
            pdf_path: Path to the PDF file
            document_id: Optional override for the document ID used in sentence IDs.
                        If not provided, derives from filename (pdf_path stem).
                        This is important when using temporary files, as the actual
                        document key should be used instead of temp filename.

        Returns:
            List[Sentence] objects.
        """
        path = Path(pdf_path)
        
        # Use provided document_id or derive from filename
        doc_id = document_id if document_id else path.stem
        
        if not path.exists():
            return []


        try:
            extracted = extract_citation_metadata(path)
        except Exception as e:
            logger.debug("Citation extraction failed for %s: %s", path, e)
            return []

        out_sentences = []
        for i,(sentence,metadata) in enumerate(extracted.items()):
            out = Sentence(
                id=f"{doc_id}_sent_{i}",
                document_id=doc_id,
                page=metadata.page,
                page_section=0,
                sentence_index=i,
                text=sentence,
                is_embedded=False,
                citation_numbers=metadata.citation_numbers,
                referenced_texts=metadata.referenced_texts,
                referenced_bibtex=metadata.referenced_bibtex,
            )
            out_sentences.append(out)

        return out_sentences

    def _normalize_ws(self, s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

def process_document(document: Document) -> List[Sentence]:
    """Process a document to extract sentences."""
    processor = PDFProcessor()
    return processor.extract_quarter_sections(str(document.pdf_path))
