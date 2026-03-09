import sys
import os

# Add src to path like main.py does
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from semtero.citation_extractor import (
    extract_citation_numbers_from_sentence,
    reference_text_to_bibtex,
)


def test_extract_citation_numbers_from_sentence_simple():
    s = "Most models have an encoder-decoder structure [5, 2, 35]."
    assert extract_citation_numbers_from_sentence(s) == [2, 5, 35]


def test_extract_citation_numbers_from_sentence_ranges_and_invalid_ignored():
    s = "See [7-9, 12] and also [not a citation] and [3; 4]."
    assert extract_citation_numbers_from_sentence(s) == [3, 4, 7, 8, 9, 12]


def test_reference_text_to_bibtex_minimal():
    ref = "Ashish Vaswani, Noam Shazeer, and Niki Parmar. Attention is all you need. In Advances in Neural Information Processing Systems, 2017."
    bib = reference_text_to_bibtex(ref, number=5)
    assert bib is not None
    assert bib.startswith("@")
    assert "title" in bib.lower()
    assert "author" in bib.lower()
    assert "2017" in bib


def test_extract_citation_numbers_from_sentence_mid_sentence_multiple_groups():
    s = (
        "In the following sections, we will describe the Transformer, motivate self-attention "
        "and discuss its advantages over models such as [17, 18] and [9]."
    )
    assert extract_citation_numbers_from_sentence(s) == [9, 17, 18]
