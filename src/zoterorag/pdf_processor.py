"""PDF processor using pymupdf4llm for text extraction and sectioning."""

import re
import contextlib
import io
import logging
from pathlib import Path
from typing import List, Generator

import pymupdf4llm

from .config import config
from .models import Document, Section, SentenceWindow

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Process PDFs to extract sections and sentence windows."""

    # Regex patterns for markdown headings
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(self, use_layout: bool = True, page_splits: int | None = None):
        """Initialize PDF processor.
        
        Args:
            use_layout: If True, use layout-preserving extraction for better
                       page structure analysis (headings, tables, etc.)
            page_splits: Number of sections to split each page into (default from config)
        """
        self.use_layout = use_layout
        self.page_splits = page_splits if page_splits is not None else config.PAGE_SPLITS

    def extract_markdown(self, pdf_path: str) -> str:
        """Extract markdown text from PDF using pymupdf4llm.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted markdown text
        """
        path = Path(pdf_path)
        if not path.exists():
            return ""

        try:
            # pymupdf4llm will sometimes emit noisy warnings like:
            # "Warning - arguments ignored in legacy mode: {...}"
            # We silence those by capturing stdout/stderr during extraction.
            buf_out = io.StringIO()
            buf_err = io.StringIO()
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                # Use layout-preserving extraction for better page structure
                if self.use_layout:
                    md_text = pymupdf4llm.to_markdown(
                        str(path),
                        page_chunks=True,
                        # Layout preservation options
                        write_images=False,
                        extract_images=False,
                    )
                else:
                    md_text = pymupdf4llm.to_markdown(str(path))

            # Handle both string and list returns
            if isinstance(md_text, list):
                return "\n".join(chunk.get("text", "") for chunk in md_text)
            return str(md_text) if md_text else ""
        except Exception as e:
            logger.error("Error extracting from %s: %s", pdf_path, e)
            return ""

    def _find_headings_in_text(
        self, 
        text: str, 
        prev_text: str, 
        prev_title: str, 
        prev_level: int,
        start_page: int,
        current_page: int,
        section_index: int,
        doc_id: str
    ) -> list[Section]:
        """Find headings in text and create Section objects."""
        sections = []
        
        # Add accumulated previous text as a section if non-empty
        if prev_text.strip():
            sections.append(Section(
                id=f"{doc_id}_{section_index}",
                document_id=doc_id,
                title=prev_title,
                level=prev_level,
                start_page=start_page,
                end_page=max(start_page, current_page - 1),
                text=prev_text.strip(),
            ))
            section_index += 1

        # Find all headings in the text
        lines = text.split("\n")
        current_section_text = ""
        current_title = prev_title
        current_level = prev_level
        
        for line in lines:
            heading_match = self.HEADING_PATTERN.match(line)
            if heading_match:
                # Save previous section if non-empty
                if current_section_text.strip():
                    sections.append(Section(
                        id=f"{doc_id}_{section_index}",
                        document_id=doc_id,
                        title=current_title,
                        level=current_level,
                        start_page=start_page,
                        end_page=max(start_page, current_page - 1),
                        text=current_section_text.strip(),
                    ))
                    section_index += 1

                # Start new section
                hashes, title = heading_match.groups()
                current_title = title.strip()
                current_level = len(hashes)
                current_section_text = ""
            else:
                current_section_text += " " + line

        return sections

    def _remove_headings(self, text: str) -> str:
        """Remove heading lines from text."""
        lines = []
        for line in text.split("\n"):
            if not self.HEADING_PATTERN.match(line):
                lines.append(line)
        return "\n".join(lines)

    # ---------- Markdown Sanitization Methods ----------

    def sanitize_markdown(self, text: str) -> str:
        """Convert markdown-formatted text to plain text.
        
        Strips all markdown annotations while preserving readable content.
        
        Args:
            text: Text with potential markdown formatting
            
        Returns:
            Plain text without markdown annotations
        """
        if not text:
            return ""
        
        # Apply sanitization steps in order
        text = self._strip_heading_markers(text)  # Remove heading markers (#), keep content
        text = self._remove_images(text)
        text = self._convert_links_to_text(text)
        text = self._extract_code_from_fenced_blocks(text)  # Extract code content before removing fences
        text = self._strip_code_formatting(text)  # Remove inline code markers
        text = self._strip_bold_italic(text)
        text = self._cleanup_blockquotes(text)
        text = self._remove_horizontal_rules(text)
        text = self._cleanup_lists(text)
        text = self._normalize_whitespace(text)
        
        return text.strip()

    def _strip_heading_markers(self, text: str) -> str:
        """Remove markdown heading markers (#), keeping the content."""
        # Remove leading # and space from ATX-style headings
        lines = []
        for line in text.split("\n"):
            # Match lines starting with 1-6 hash symbols followed by space, keep rest of line
            stripped = re.sub(r'^#{1,6}\s+', '', line)
            lines.append(stripped)
        return "\n".join(lines)

    def _extract_code_from_fenced_blocks(self, text: str) -> str:
        """Extract content from fenced code blocks while removing the fences."""
        # Pattern matches ```language...content...``` (non-greedy)
        result = []
        in_code_block = False
        lines = text.split("\n")
        
        for line in lines:
            if re.match(r'^```', line):
                # Toggle state - don't include the fence line itself
                in_code_block = not in_code_block
                continue
            
            if in_code_block:
                # Inside code block - keep the content
                result.append(line)
            else:
                result.append(line)
        
        return "\n".join(result)

    def _remove_images(self, text: str) -> str:
        """Remove image references entirely."""
        # ![alt](url) or ![alt](url "title")
        return re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)

    def _convert_links_to_text(self, text: str) -> str:
        """Convert links to just the link text."""
        # [text](url) -> text
        return re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    def _strip_code_formatting(self, text: str) -> str:
        """Strip inline code markers (fenced blocks handled separately)."""
        # Note: Fenced code block content is now preserved via _extract_code_from_fenced_blocks
        # This only removes inline code (`code`)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        return text

    def _strip_bold_italic(self, text: str) -> str:
        """Remove bold and italic markers."""
        # **bold** or __bold__
        text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        # *italic* or _italic_
        text = re.sub(r'(?<!\w)\*([^\*]+)\*(?!\w)', r'\1', text)
        text = re.sub(r'(?<![a-zA-Z])_([^_]+)_(?![a-zA-Z])', r'\1', text)
        return text

    def _cleanup_blockquotes(self, text: str) -> str:
        """Convert blockquotes to plain text."""
        # > quote or >>> nested
        lines = []
        for line in text.split("\n"):
            # Remove leading > markers (including nested >)
            cleaned = re.sub(r'^>+\s*', '', line)
            lines.append(cleaned)
        return "\n".join(lines)

    def _remove_horizontal_rules(self, text: str) -> str:
        """Remove horizontal rule lines."""
        # ---, ***, ___ on their own lines
        text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        return text

    def _cleanup_lists(self, text: str) -> str:
        """Convert list markers to plain text items."""
        lines = []
        for line in text.split("\n"):
            # Match -, *, +, or numbered lists at start of line
            cleaned = re.sub(r'^[\s]*([-*+]|\d+\.)\s+', '', line)
            lines.append(cleaned)
        return "\n".join(lines)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph breaks."""
        # Replace multiple blank lines with single blank line
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Replace multiple spaces with single space within lines
        text = re.sub(r' +', ' ', text)
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines)

    def extract_quarter_sections(self, pdf_path: str | Path) -> List[Section]:
        """Extract sections by splitting each page into equal parts based on line count.
        
        Uses pymupdf4llm to get page chunks, then splits lines within each page
        into N sections (default 4) for more granular embeddings.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Section objects, one per split portion per page
        """
        path = Path(pdf_path)
        if not path.exists():
            return []

        try:
            # Get page-level chunks with text content.
            # Capture stdout/stderr to suppress noisy legacy-mode warnings.
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
            # Single page or error - treat as one section
            text = str(md_text) if md_text else ""
            lines = [l for l in text.split("\n") if l.strip()]
            if lines:
                # Sanitize markdown formatting from extracted text
                sanitized_text = self.sanitize_markdown(text)
                return [Section(
                    id=f"{path.stem}_sec_0",
                    document_id=path.stem,
                    title="Page 1",
                    level=1,
                    start_page=1,
                    end_page=1,
                    text=sanitized_text.strip(),
                    page_section=None,  # No splitting applied
                )]
            return []

        sections: List[Section] = []
        section_index = 0

        for chunk in md_text:
            page = chunk.get("page", 1)
            text = chunk.get("text", "")
            
            # Split into lines and filter empty ones
            lines = [l for l in text.split("\n") if l.strip()]
            
            if not lines:
                continue
            
            # Calculate split size based on number of splits per page
            num_splits = self.page_splits
            lines_per_split = max(1, len(lines) // num_splits)
            
            for i in range(num_splits):
                start_idx = i * lines_per_split
                end_idx = min((i + 1) * lines_per_split, len(lines))
                
                # Last split gets remaining lines
                if i == num_splits - 1:
                    end_idx = len(lines)
                
                section_lines = lines[start_idx:end_idx]
                if not section_lines:
                    continue
                
                section_text = "\n".join(section_lines)
                
                # Sanitize markdown formatting from extracted text
                sanitized_text = self.sanitize_markdown(section_text)
                
                sections.append(Section(
                    id=f"{path.stem}_sec_{section_index}",
                    document_id=path.stem,
                    title=f"Page {page} Section {i + 1}",
                    level=1,
                    start_page=page,
                    end_page=page,
                    text=sanitized_text.strip(),
                    page_section=i + 1,  # 1-indexed position within the page
                ))
                section_index += 1

        return sections

    def create_sentence_windows(
        self, 
        section: Section, 
        window_size: int = 3,
        overlap: int = 2
    ) -> List[SentenceWindow]:
        """Create sliding windows of sentences from section text.
        
        Creates overlapping windows where each subsequent window shares
        (window_size - overlap) sentences with the previous one.
        
        Args:
            section: The section to create windows from
            window_size: Number of sentences per window (default 3)
            overlap: Number of sentences that overlap between windows (default 2,
                     meaning windows slide by 1 sentence each time for maximum overlap)
        
        Returns:
            List of SentenceWindow objects with overlapping content.
        """
        # Split into sentences using common delimiters
        sentence_list = re.split(r"(?<=[.!?])\s+", section.text)
        sentence_list = [s.strip() for s in sentence_list if s.strip()]
        
        windows: List[SentenceWindow] = []
        step = max(1, window_size - overlap)  # How many sentences to advance
        
        for i in range(0, len(sentence_list) - window_size + 1, step):
            window_sentences = sentence_list[i:i + window_size]
            window_text = " ".join(window_sentences)
            
            # Create ID from section_id and index
            window_id = f"{section.id}_win_{i}"
            
            windows.append(SentenceWindow(
                id=window_id,
                section_id=section.id,
                window_index=i,
                sentences=window_sentences,
                text=window_text,
                is_embedded=False
            ))
        
        # Handle case where we have some sentences but less than full window_size
        if not windows and sentence_list:
            windows.append(SentenceWindow(
                id=f"{section.id}_win_0",
                section_id=section.id,
                window_index=0,
                sentences=sentence_list,
                text=section.text,
                is_embedded=False
            ))
            
        return windows


def process_document(document: Document) -> List[Section]:
    """Process a document to extract its sections.
    
    Uses line-count based splitting (extract_quarter_sections) for more
    granular embeddings rather than heading-based sectioning.
    """
    processor = PDFProcessor()
    return processor.extract_quarter_sections(str(document.pdf_path))
