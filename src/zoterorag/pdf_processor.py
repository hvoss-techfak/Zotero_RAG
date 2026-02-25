"""PDF processor using pymupdf4llm for text extraction and sectioning."""

import re
from pathlib import Path
from typing import List, Generator

import pymupdf4llm

from .config import config
from .models import Document, Section, SentenceWindow


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
            print(f"Error extracting from {pdf_path}: {e}")
            return ""

    def segment_into_sections(self, markdown: str, document_id: str) -> List[Section]:
        """Segment markdown text into sections based on headings."""
        lines = markdown.split("\n")
        sections: List[Section] = []
        
        current_section_text = ""
        current_title = "Introduction"
        current_level = 1
        section_index = 0
        
        for line in lines:
            heading_match = self.HEADING_PATTERN.match(line)
            if heading_match:
                # Save previous section if non-empty
                if current_section_text.strip():
                    sections.append(Section(
                        id=f"{document_id}_sec_{section_index}",
                        document_id=document_id,
                        title=current_title,
                        level=current_level,
                        start_page=1,
                        end_page=1,
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

        # Add final section if there's remaining text
        if current_section_text.strip():
            sections.append(Section(
                id=f"{document_id}_sec_{section_index}",
                document_id=document_id,
                title=current_title,
                level=current_level,
                start_page=1,
                end_page=1,
                text=current_section_text.strip(),
            ))

        return sections

    def extract_sections(self, pdf_path: Path) -> list[Section]:
        """Extract sections from a PDF file based on headings."""
        if not pdf_path.exists():
            return []

        # Use pymupdf4llm to extract markdown
        try:
            md_text = pymupdf4llm.to_markdown(
                str(pdf_path),
                page_chunks=True,  # Get page-level chunks for page tracking
            )
        except Exception as e:
            print(f"Error extracting from {pdf_path}: {e}")
            return []

        sections = []
        current_section_text = ""
        current_title = "Introduction"
        current_level = 1
        start_page = 1
        
        # Track which page we're on
        current_page = 1

        if isinstance(md_text, list):
            # Page chunks mode - process each chunk
            for i, chunk in enumerate(md_text):
                text = chunk.get("text", "")
                page = chunk.get("page", i + 1)
                
                # Find headings in this chunk
                sections_from_chunk = self._find_headings_in_text(
                    text, current_section_text, current_title, current_level,
                    start_page, page, len(sections), pdf_path.stem
                )
                
                for sec in sections_from_chunk:
                    if sec.text.strip():  # Only add non-empty sections
                        sections.append(sec)
                    
                    # Update current tracking
                    current_section_text = ""
                    current_title = sec.title
                    current_level = sec.level
                    start_page = page + 1

                # Accumulate non-heading text as part of current section
                non_heading = self._remove_headings(text)
                if non_heading.strip():
                    current_section_text += " " + non_heading
        else:
            # Single string mode - find headings in full text
            sections_from_text = self._find_headings_in_text(
                md_text, "", "Introduction", 1, 1, 1, len(sections), pdf_path.stem
            )
            for sec in sections_from_text:
                if sec.text.strip():
                    sections.append(sec)

        # Add final section if there's remaining text
        if current_section_text.strip():
            sections.append(Section(
                id=f"{pdf_path.stem}_{len(sections)}",
                document_id=pdf_path.stem,
                title=current_title,
                level=current_level,
                start_page=start_page,
                end_page=current_page,
                text=current_section_text.strip(),
            ))

        return sections

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
            # Get page-level chunks with text content
            md_text = pymupdf4llm.to_markdown(
                str(path),
                page_chunks=True,
                write_images=False,
                extract_images=False,
            )
        except Exception as e:
            print(f"Error extracting from {pdf_path}: {e}")
            return []

        if not isinstance(md_text, list):
            # Single page or error - treat as one section
            text = str(md_text) if md_text else ""
            lines = [l for l in text.split("\n") if l.strip()]
            if lines:
                return [Section(
                    id=f"{path.stem}_sec_0",
                    document_id=path.stem,
                    title="Page 1",
                    level=1,
                    start_page=1,
                    end_page=1,
                    text=text.strip(),
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
                
                sections.append(Section(
                    id=f"{path.stem}_sec_{section_index}",
                    document_id=path.stem,
                    title=f"Page {page} Section {i + 1}",
                    level=1,
                    start_page=page,
                    end_page=page,
                    text=section_text.strip(),
                    page_section=i + 1,  # 1-indexed position within the page
                ))
                section_index += 1

        return sections

    def create_sentence_windows(self, section: Section, window_size: int = 3) -> List[SentenceWindow]:
        """Create sliding windows of sentences from section text.
        
        Returns list of SentenceWindow objects.
        """
        # Split into sentences using common delimiters
        sentence_list = re.split(r"(?<=[.!?])\s+", section.text)
        sentence_list = [s.strip() for s in sentence_list if s.strip()]
        
        windows: List[SentenceWindow] = []
        for i in range(len(sentence_list) - window_size + 1):
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
        
        # If fewer sentences than window_size, still create one window with all text
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
    """Process a document to extract its sections."""
    processor = PDFProcessor()
    markdown = processor.extract_markdown(str(document.pdf_path))
    return processor.segment_into_sections(markdown, document.zotero_key)
