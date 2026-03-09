"""Citation extraction and bibliography parsing for scientific PDFs.

We target the common numeric bracket citation style used in many arXiv papers:
    "... encoder-decoder structure [5, 2, 35]"
with a References section like:
    "[5] Author. Title. Venue, 2016."

The main entrypoint for integration is :func:`extract_citation_metadata`, which
returns a mapping of sentence text -> resolved citation metadata. The rest of the
app stores these fields alongside sentence embeddings.

This module uses pymupdf4llm to extract text consistently with pdf_processor.py,
ensuring that sentence boundaries match when attaching citations.
"""

from __future__ import annotations

import io
import contextlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pymupdf4llm

_CITATION_GROUP_RE = re.compile(r"\[(?P<body>[^]]{1,80})]")
_CITATION_BODY_OK_RE = re.compile(
    r"^\s*\d+(?:\s*[-–]\s*\d+)?(?:\s*[,;]\s*\d+(?:\s*[-–]\s*\d+)?)*\s*$"
)


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _expand_numbers(body: str) -> list[int]:
    """Parse something like '5, 2, 35' or '7-9, 12' into a sorted list."""

    body = (body or "").replace("–", "-")
    out: set[int] = set()
    for part in re.split(r"\s*[,;]\s*", body.strip()):
        if not part:
            continue
        if "-" in part:
            a, b = [p.strip() for p in part.split("-", 1)]
            if a.isdigit() and b.isdigit():
                start, end = int(a), int(b)
                if start <= end and end - start <= 2000:
                    out.update(range(start, end + 1))
            continue
        if part.strip().isdigit():
            out.add(int(part.strip()))
    return sorted(out)


def extract_citation_numbers_from_sentence(sentence: str) -> list[int]:
    """Return sorted unique citation numbers for a single sentence."""

    numbers: set[int] = set()
    for m in _CITATION_GROUP_RE.finditer(sentence or ""):
        body = m.group("body")
        if not _CITATION_BODY_OK_RE.match(body):
            continue
        numbers.update(_expand_numbers(body))
    return sorted(numbers)


def extract_page_text_from_pymupdf4llm(pdf_path: str | Path) -> list[dict]:
    """Extract page text using pymupdf4llm for consistent extraction.

    Returns a list of dicts with 'page' (1-based) and 'text' keys.
    This matches the format used by pdf_processor.py.
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

        if isinstance(md_text, list):
            return [
                {"page": chunk.get("page", 1), "text": chunk.get("text", "")}
                for chunk in md_text
            ]
        elif md_text:
            return [{"page": 1, "text": str(md_text)}]
        return []
    except Exception:
        return []


def find_references_start_page(pages: list[dict]) -> int:
    """Return 0-based page index for the start of references section."""

    heading_re = re.compile(r"^\s*References\s*$", re.I | re.M)

    for i, chunk in enumerate(pages):
        text = chunk.get("text", "")
        if heading_re.search(text):
            return i

    refstart_re = re.compile(r"^\s*\[(\d{1,4})]\s+", re.M)
    # Check last 8 pages for reference-like numbering
    start_check = max(0, len(pages) - 8)
    for i in range(start_check, len(pages)):
        text = pages[i].get("text", "")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        starts = sum(1 for ln in lines if refstart_re.match(ln))
        if lines and starts / len(lines) > 0.25:
            return i

    return max(0, len(pages) - 1)


def parse_references_from_pages(
    pages: list[dict], start_page_idx: int
) -> dict[int, str]:
    """Parse references of the form '[12] ...' into a mapping."""

    ref_line_re = re.compile(r"^\s*\[(\d{1,4})]\s+(.*)$")

    entries: dict[int, str] = {}
    cur_n: Optional[int] = None
    cur_text = ""

    def flush() -> None:
        nonlocal cur_n, cur_text
        if cur_n is not None:
            t = _normalize_ws(cur_text)
            if t:
                entries[cur_n] = t
        cur_n = None
        cur_text = ""

    for i in range(start_page_idx, len(pages)):
        page_text = pages[i].get("text", "")
        for raw_line in page_text.splitlines():
            line = _normalize_ws(raw_line)
            if not line:
                continue
            if line.isdigit():
                continue
            if re.match(r"^\s*References\s*$", line, re.I):
                continue

            m = ref_line_re.match(line)
            if m:
                flush()
                cur_n = int(m.group(1))
                cur_text = m.group(2)
                continue

            if cur_n is not None:
                if cur_text.endswith("-") and line and line[0].isalpha():
                    cur_text = cur_text[:-1] + line
                else:
                    cur_text += " " + line

    flush()
    return entries


def _bibtex_key_from_authors_year(authors: list[str], year: Optional[str]) -> str:
    last = "ref"
    if authors:
        first = authors[0].strip()
        if "," in first:
            last = first.split(",", 1)[0].strip()
        else:
            last = first.split()[-1].strip()
        last = re.sub(r"[^A-Za-z0-9]", "", last) or "ref"
    return f"{last}{year or ''}".lower()


def _split_authors(authors_raw: str) -> list[str]:
    s = (authors_raw or "").strip().rstrip(".")
    if not s:
        return []

    s = s.replace(" & ", " and ")
    s = re.sub(r",\s+and\s+", " and ", s)

    comma_parts = [p.strip() for p in re.split(r",\s+", s) if p.strip()]
    out: list[str] = []
    for part in comma_parts:
        part = re.sub(r"^and\s+", "", part.strip(), flags=re.I)
        if part:
            out.append(part)

    return [p for p in out if p.lower() not in {"et al", "et al."}]


def reference_text_to_bibtex(
    ref_text: str, number: Optional[int] = None
) -> Optional[str]:
    """Convert a reference string into a minimal BibTeX entry (heuristic)."""

    txt = _normalize_ws(ref_text)
    if not txt:
        return None

    doi_m = re.search(r"\b(10\.\d{4,9}/[^\s]+)\b", txt)
    arxiv_m = re.search(r"\barXiv:(\d{4}\.\d{4,5})(?:v\d+)?\b", txt, re.I)

    years = re.findall(r"\b(19\d{2}|20\d{2})\b", txt)
    year = years[-1] if years else None

    parts = [p.strip() for p in txt.split(".") if p.strip()]
    if len(parts) < 2:
        return None

    authors_raw = parts[0]
    title = parts[1]
    authors = _split_authors(authors_raw)

    title = re.sub(r"\s+In\s+.+$", "", title).strip().rstrip(",")

    booktitle: Optional[str] = None
    journal: Optional[str] = None
    publisher: Optional[str] = None
    pages: Optional[str] = None

    pages_m = re.search(r"\bpages\s+([0-9]+\s*[–-]\s*[0-9]+)\b", txt, re.I)
    if pages_m:
        pages = pages_m.group(1).replace(" ", "")

    in_m = re.search(
        r"\bIn\s+(.+?)(?:,\s*pages\s+[0-9]|,\s*(?:ACL|IEEE|Springer|AAAI)|,\s*(?:August|June|July)\s+\d{4}|,\s*\d{4}|\.|$)",
        txt,
    )
    if in_m:
        booktitle = _normalize_ws(in_m.group(1)).rstrip(",")
        booktitle = re.sub(r"\bProc\.\s+of\b", "Proceedings of", booktitle)
    else:
        if re.search(r"\bCoRR\b", txt):
            journal = "CoRR"

    pub_m = re.search(r"\.\s*(ACL|Curran Associates, Inc\.|IEEE)\b", txt)
    if pub_m:
        publisher = pub_m.group(1)

    if arxiv_m or ("arXiv" in txt and not booktitle and not journal):
        entry_type = "misc"
    elif booktitle:
        entry_type = "inproceedings"
    else:
        entry_type = "article"

    key = _bibtex_key_from_authors_year(authors, year)
    if number is not None:
        key = f"{key}{number}"

    fields: list[tuple[str, str]] = []
    if authors:
        fields.append(("author", " and ".join(authors)))
    if title:
        fields.append(("title", title))
    if year:
        fields.append(("year", year))

    if entry_type == "inproceedings" and booktitle:
        fields.append(("booktitle", booktitle))
    if entry_type == "article" and journal:
        fields.append(("journal", journal))

    if pages:
        fields.append(("pages", pages))
    if publisher:
        fields.append(("publisher", publisher))

    if doi_m:
        fields.append(("doi", doi_m.group(1).rstrip(".,;")))

    if arxiv_m:
        fields.append(("eprint", arxiv_m.group(1)))
        fields.append(("archivePrefix", "arXiv"))
        cls_m = re.search(r"\barXiv:\d{4}\.\d{4,5}(?:v\d+)?\s*\[([^]]+)]", txt)
        if cls_m:
            fields.append(("primaryClass", cls_m.group(1).strip()))

    have_author = any(k == "author" for k, _ in fields)
    have_title = any(k == "title" for k, _ in fields)
    if not (have_author and have_title):
        return None

    def esc(v: str) -> str:
        return v.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

    body = ",\n  ".join(f"{k} = {{{esc(v)}}}" for k, v in fields)
    return f"@{entry_type}{{{key},\n  {body}\n}}"


@dataclass(frozen=True)
class CitationMetadata:
    page: int
    citation_numbers: list[int]
    referenced_texts: list[str]
    referenced_bibtex: list[str]


def extract_citation_metadata(pdf_path: str | Path) -> dict[str, CitationMetadata]:
    """Extract citation metadata keyed by *exact* sentence text.

    Uses pymupdf4llm for consistent text extraction with pdf_processor.py.
    This ensures sentence boundaries match when attaching citations.
    """

    # Use pymupdf4llm to extract pages consistently
    pages = extract_page_text_from_pymupdf4llm(pdf_path)
    if not pages:
        return {}

    ref_start_idx = find_references_start_page(pages)
    refs = parse_references_from_pages(pages, ref_start_idx)
    refs_bibtex: dict[int, Optional[str]] = {
        n: reference_text_to_bibtex(t, number=n) for n, t in refs.items()
    }

    # Use same splitter as pdf_processor.py for consistent sentence boundaries
    splitter = re.compile(r"(?<=[.!?])\s+(?=[\"\[(]?[A-Z0-9])")

    sentence_map: dict[str, CitationMetadata] = {}

    # Process pages before the references section
    for i in range(0, max(0, ref_start_idx)):
        chunk = pages[i]
        text = chunk.get("text", "")

        if not text:
            continue

        # Split into lines and filter out standalone page numbers
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        filtered = [ln for ln in lines if not (ln.isdigit() and len(ln) <= 3)]

        combined_text = _normalize_ws(" ".join(filtered))
        if not combined_text:
            continue

        # Split into sentences using the same pattern as pdf_processor.py
        for sent in splitter.split(combined_text):
            s = _normalize_ws(sent)

            if (
                not s or len(s.split()) <= 3
            ):  # Skip short sentences like in pdf_processor
                continue

            nums = extract_citation_numbers_from_sentence(s)

            referenced_texts: list[str] = []
            referenced_bibtex: list[str] = []
            for n in nums:
                if n in refs:
                    referenced_texts.append(refs[n])
                    b = refs_bibtex.get(n)
                    if b:
                        referenced_bibtex.append(b)
            sentence_map[s] = CitationMetadata(
                page=chunk.get("page", 1),
                citation_numbers=nums,
                referenced_texts=referenced_texts,
                referenced_bibtex=referenced_bibtex,
            )

    return sentence_map
