"""Tests for PDF processor markdown sanitization."""

import sys
import os

# Add src to path like main.py does
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from zoterorag.pdf_processor import PDFProcessor


class TestSanitizeMarkdown:
    """Test suite for markdown sanitization."""

    @pytest.fixture
    def processor(self):
        return PDFProcessor()

    # ----- Bold/Italic Tests -----

    def test_bold_double_asterisk(self, processor):
        text = "This is **bold** text"
        result = processor.sanitize_markdown(text)
        assert result == "This is bold text"

    def test_bold_double_underscore(self, processor):
        text = "This is __bold__ text"
        result = processor.sanitize_markdown(text)
        assert result == "This is bold text"

    def test_italic_single_asterisk(self, processor):
        text = "This is *italic* text"
        result = processor.sanitize_markdown(text)
        assert result == "This is italic text"

    def test_italic_single_underscore_word_boundary(self, processor):
        # Underscores between words should become italic markers
        text = "This is _italic_ text"
        result = processor.sanitize_markdown(text)
        assert result == "This is italic text"

    def test_bold_italic_combined(self, processor):
        text = "**Bold and *italic* together**"
        # Note: order matters - bold stripped first
        result = processor.sanitize_markdown(text)
        assert "Bold" in result

    # ----- Link Tests -----

    def test_link_simple(self, processor):
        text = "Visit [Google](https://google.com) for more info"
        result = processor.sanitize_markdown(text)
        assert result == "Visit Google for more info"

    def test_link_with_title(self, processor):
        text = 'Click [here](https://example.com "Example Site")'
        result = processor.sanitize_markdown(text)
        assert result == "Click here"

    # ----- Image Tests -----

    def test_image_simple(self, processor):
        text = "Some text ![alt text](image.png) more text"
        result = processor.sanitize_markdown(text)
        # Image is removed entirely (no placeholder), whitespace normalized
        assert result == "Some text more text"

    def test_image_with_title(self, processor):
        text = '![Logo](logo.png "My Logo")'
        result = processor.sanitize_markdown(text)
        assert result == ""

    # ----- Code Tests -----

    def test_inline_code(self, processor):
        text = "Use `print()` function to output"
        result = processor.sanitize_markdown(text)
        assert result == "Use print() function to output"

    def test_fenced_code_block(self, processor):
        text = """Here is code:
```python
def hello():
    print("Hello")
```
End of code"""
        result = processor.sanitize_markdown(text)
        # Fence markers removed but content preserved
        assert "def hello" in result
        assert "print" in result
        assert "```" not in result  # Code fences should be removed

    def test_fenced_code_block_multiline(self, processor):
        text = """```
line1
line2
```
"""
        result = processor.sanitize_markdown(text)
        # Fence markers removed but content preserved (now extracted like other content)
        assert "line1" in result
        assert "line2" in result

    # ----- Blockquote Tests -----

    def test_blockquote_simple(self, processor):
        text = "> This is a quote"
        result = processor.sanitize_markdown(text)
        assert result == "This is a quote"

    def test_blockquote_nested(self, processor):
        text = """> Level 1
>> Level 2
>>> Level 3"""
        result = processor.sanitize_markdown(text)
        assert "Level 1" in result
        assert "Level 2" in result
        assert "Level 3" in result

    # ----- Horizontal Rule Tests -----

    def test_horizontal_rule_dashes(self, processor):
        text = "Some text\n---\nMore text"
        result = processor.sanitize_markdown(text)
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) == 2
        assert "Some text" in lines[0]

    def test_horizontal_rule_stars(self, processor):
        text = "Text\n***\nMore text"
        result = processor.sanitize_markdown(text)
        assert "Text" in result
        assert "More text" in result

    # ----- List Tests -----

    def test_unordered_list_dash(self, processor):
        text = """- Item one
- Item two
- Item three"""
        result = processor.sanitize_markdown(text)
        lines = [l.strip() for l in result.split("\n") if l.strip()]
        assert "Item one" in lines[0]
        assert "Item two" in lines[1]

    def test_unordered_list_asterisk(self, processor):
        text = """* First item
* Second item"""
        result = processor.sanitize_markdown(text)
        assert "First item" in result

    def test_ordered_list(self, processor):
        text = """1. First step
2. Second step
3. Third step"""
        result = processor.sanitize_markdown(text)
        lines = [l.strip() for l in result.split("\n") if l.strip()]
        assert "First step" in lines[0]
        assert "Second step" in lines[1]

    def test_list_with_subitems(self, processor):
        text = """- Item A
  - Subitem A1
  - Subitem A2
- Item B"""
        result = processor.sanitize_markdown(text)
        assert "Item A" in result
        assert "Subitem A1" in result

    # ----- Whitespace Tests -----

    def test_multiple_spaces_normalized(self, processor):
        text = "Word1     Word2    Word3"
        result = processor.sanitize_markdown(text)
        assert "Word1 Word2 Word3" == result
        assert "     " not in result

    def test_multiple_blank_lines_normalized(self, processor):
        text = "Paragraph 1\n\n\n\nParagraph 2"
        result = processor.sanitize_markdown(text)
        # Multiple newlines should become max 2
        assert "\n\n\n" not in result

    # ----- Edge Cases -----

    def test_empty_string(self, processor):
        result = processor.sanitize_markdown("")
        assert result == ""

    def test_none_input(self, processor):
        result = processor.sanitize_markdown(None)
        assert result == ""

    def test_no_markdown_text(self, processor):
        text = "This is just plain text without any markdown."
        result = processor.sanitize_markdown(text)
        assert result == "This is just plain text without any markdown."

    def test_complex_markdown_document(self, processor):
        text = """# Heading

Some **bold** and *italic* text.

- List item 1
- List item 2

> A blockquote here

[Link](https://example.com)

```
code block
```

---
"""
        result = processor.sanitize_markdown(text)

        # Should contain expected content (heading markers removed, content kept)
        assert "Heading" in result
        assert "bold" in result
        assert "italic" in result
        assert "List item 1" in result
        assert "blockquote here" in result
        assert "Link" in result
        # Heading marker (#) should be removed from heading lines but could appear elsewhere
        assert "# Heading" not in result
        assert "**" not in result  # Bold markers gone

    def test_realistic_pdf_extraction(self, processor):
        """Test with realistic markdown that might come from PDF extraction."""
        text = """
## Machine Learning Introduction

Machine learning is a **subset** of artificial intelligence.

### Types of ML

1. *Supervised* Learning
2. Unsupervised Learning
3. Reinforcement Learning

> Important: ML models require training data!

![diagram](ml-diagram.png)

See also: [Wikipedia](https://en.wikipedia.org/wiki/Machine_learning)
"""
        result = processor.sanitize_markdown(text)

        # Check key content is preserved
        assert "Machine learning" in result or "machine learning" in result.lower()
        assert "subset" in result  # bold stripped from "**subset**"
        assert "Supervised" in result or "supervised" in result.lower()
        assert "training data" in result  # blockquote content preserved
        assert "Wikipedia" in result  # link text kept, URL removed


class TestSanitizeMarkdownHelperMethods:
    """Test individual helper methods."""

    @pytest.fixture
    def processor(self):
        return PDFProcessor()

    def test_remove_images_exact_match(self, processor):
        assert processor._remove_images("text ![img](url) more") == "text  more"

    def test_convert_links_preserves_text(self, processor):
        assert (
            processor._convert_links_to_text("[click me](http://x.com)") == "click me"
        )

    def test_strip_code_fencing_removes_backticks(self, processor):
        text = "Use `code` for inline"
        result = processor._strip_code_formatting(text)
        assert "`" not in result

    def test_cleanup_lists_various_formats(self, processor):
        text = "- a\n* b\n+ c\n1. d"
        result = processor._cleanup_lists(text)
        lines = [l for l in result.split("\n") if l.strip()]
        # All list markers should be removed
        assert all(not l.startswith(("-", "*", "+", "1.")) for l in lines)
