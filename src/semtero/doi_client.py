"""Helpers for resolving DOI metadata.

Currently this module is focused on fetching BibTeX from doi.org.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import requests


_DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)


def normalize_doi(doi: str) -> str:
    """Normalize an input DOI.

    Accepts:
      - Bare DOI: "10.1111/cgf.13217"
      - DOI URL: "https://doi.org/10.1111/cgf.13217"
      - dx.doi.org URL

    Returns the bare DOI string.

    Raises:
        ValueError: if no DOI can be parsed.
    """
    if not doi or not doi.strip():
        raise ValueError("DOI must be a non-empty string")

    s = doi.strip()
    m = _DOI_RE.search(s)
    if not m:
        raise ValueError(f"Could not parse DOI from input: {doi!r}")
    return m.group(0)


@dataclass(slots=True)
class DoiClient:
    """Client for fetching BibTeX from doi.org (or compatible endpoint)."""

    base_url: str = "https://doi.org/"
    timeout_seconds: float = 10.0
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        base = self.base_url or "https://doi.org/"
        object.__setattr__(self, "base_url", base.rstrip("/") + "/")

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self.session.close()

    def __enter__(self) -> "DoiClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def fetch_bibtex(self, doi: str) -> str:
        """Fetch BibTeX for a DOI.

        Uses Accept: application/x-bibtex.
        """
        norm = normalize_doi(doi)
        url = f"{self.base_url}{norm}"
        resp = self.session.get(
            url,
            headers={"Accept": "application/x-bibtex"},
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        bibtex = (resp.text or "").strip()
        if not bibtex:
            raise ValueError(f"Empty BibTeX response for DOI {norm}")
        return bibtex
