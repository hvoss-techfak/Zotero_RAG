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
            return self._sentences_from_text(
                document_id=path.stem,
                page=1,
                page_section=None,
                text=sanitized_text,
            )

        sentences: List[Sentence] = []
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

            # Re-split into sentences (defensive)
            normalized: list[str] = []
            for line in sentence_lines:
                split_sentences = re.split(r"(?<=[.!?])\s+", line)
                normalized.extend([s.strip() for s in split_sentences if s.strip()])

            if not normalized:
                continue

            num_splits = max(1, int(self.page_splits))
            lines_per_split = max(1, len(normalized) // num_splits)

            for i in range(num_splits):
                start_idx = i * lines_per_split
                end_idx = min((i + 1) * lines_per_split, len(normalized))
                if i == num_splits - 1:
                    end_idx = len(normalized)

                part_lines = normalized[start_idx:end_idx]
                if not part_lines:
                    continue

                part_text = " ".join(part_lines)
                sanitized = self.sanitize_markdown(part_text)
                if not sanitized.strip():
                    continue

                part_sentences = self._sentences_from_text(
                    document_id=path.stem,
                    page=page,
                    page_section=i + 1,
                    text=sanitized,
                    start_sentence_index=sentence_index,
                )
                sentences.extend(part_sentences)
                sentence_index += len(part_sentences)

        return sentences

    def _sentences_from_text(
        self,
        document_id: str,
        page: int,
        page_section: int | None,
        text: str,
        start_sentence_index: int = 0,
    ) -> List[Sentence]:
        sentence_list = re.split(r"(?<=[.!?])\s+", text)
        sentence_list = [s.strip() for s in sentence_list if s.strip()]

        out: List[Sentence] = []
        for i, sent in enumerate(sentence_list):
            idx = start_sentence_index + i
            out.append(
                Sentence(
                    id=f"{document_id}_sent_{idx}",
                    document_id=document_id,
                    page=page,
                    page_section=page_section,
                    sentence_index=idx,
                    text=sent,
                    is_embedded=False,
                )
            )

        return out


def process_document(document: Document) -> List[Sentence]:
    """Process a document to extract sentences."""
    processor = PDFProcessor()
    return processor.extract_quarter_sections(str(document.pdf_path))
