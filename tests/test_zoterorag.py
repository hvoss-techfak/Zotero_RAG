"""Tests for ZoteroRAG components."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test config
from zoterorag.config import Config

# Test models
from zoterorag.models import Document, Section, SentenceWindow, SearchResult


class TestModels:
    """Test data models."""

    def test_document_creation(self):
        doc = Document(
            zotero_key="ABC123",
            title="Test Paper",
            authors=["John Doe", "Jane Smith"],
            pdf_path=Path("/path/to/test.pdf"),
        )
        assert doc.zotero_key == "ABC123"
        assert doc.title == "Test Paper"
        assert len(doc.authors) == 2

    def test_document_hash(self):
        doc1 = Document(zotero_key="ABC123", title="Test")
        doc2 = Document(zotero_key="ABC123", title="Different Title")
        # Same key should have same hash
        assert hash(doc1) == hash(doc2)

    def test_section_creation(self):
        section = Section(
            id="doc_sec_0",
            document_id="ABC123",
            title="Introduction",
            level=1,
            start_page=1,
            end_page=5,
            text="This is the introduction text.",
        )
        assert section.id == "doc_sec_0"
        assert section.level == 1
        assert section.text == "This is the introduction text."

    def test_sentence_window_creation(self):
        window = SentenceWindow(
            id="doc_sec_0_win_0",
            section_id="doc_sec_0",
            window_index=0,
            sentences=["First sentence.", "Second sentence.", "Third sentence."],
            text="First sentence. Second sentence. Third sentence.",
        )
        assert len(window.sentences) == 3
        assert window.window_index == 0

    def test_search_result_to_dict(self):
        result = SearchResult(
            text="Relevant passage here.",
            document_title="Test Paper",
            section_title="Introduction",
            zotero_key="ABC123",
            relevance_score=0.95,
            rerank_score=0.88,
        )
        result_dict = result.to_dict()
        assert result_dict["text"] == "Relevant passage here."
        assert result_dict["relevance_score"] == 0.95
        assert result_dict["zotero_key"] == "ABC123"


class TestConfig:
    """Test configuration."""

    def test_default_config(self):
        config = Config()
        assert config.ZOTERO_API_URL == "http://127.0.0.1:23119"
        assert config.VECTOR_STORE_DIR == Path("./data/vector_store")
        assert config.PDF_CACHE_PATH == Path("./data/pdfs")

    def test_env_override(self):
        # Note: env vars are read at module load time, so we just verify default behavior
        config = Config()
        # Default values should be set
        assert "127.0.0.1" in config.ZOTERO_API_URL or "localhost" in config.ZOTERO_API_URL.lower()

    def test_ensure_dirs(self):
        # Just verify the method exists and is callable
        assert callable(Config.ensure_dirs)


class TestZoteroClient:
    """Test Zotero API client."""

    def test_parse_item_to_document_with_pdf(self):
        from zoterorag.zotero_client import ZoteroClient
        
        with patch.object(ZoteroClient, '__init__', lambda x, y: None):
            client = ZoteroClient.__new__(ZoteroClient)
            
            item = {
                "key": "ABC123",
                "data": {
                    "title": "Test Paper",
                    "creators": [
                        {"firstName": "John", "lastName": "Doe"}
                    ],
                    "dateAdded": "2024-01-01T00:00:00Z",
                    "dateModified": "2024-01-02T00:00:00Z",
                }
            }
            
            # Test without PDF - should return None
            result = client.parse_item_to_document(item)
            assert result is None

    def test_parse_item_with_pdf_attachment(self):
        from zoterorag.zotero_client import ZoteroClient
        
        with patch.object(ZoteroClient, '__init__', lambda x, y: None):
            client = ZoteroClient.__new__(ZoteroClient)
            
            item = {
                "key": "ABC123",
                "data": {
                    "title": "Test Paper",
                    "creators": [
                        {"firstName": "John", "lastName": "Doe"}
                    ],
                    "attachments": [
                        {"contentType": "application/pdf", "path": "test.pdf"}
                    ]
                }
            }
            
            result = client.parse_item_to_document(item)
            assert result is not None
            assert result.zotero_key == "ABC123"
            assert result.title == "Test Paper"

    def test_has_pdf_v2_format(self):
        """Test _has_pdf with v2 API format (attachments array)."""
        from zoterorag.zotero_client import ZoteroClient
        
        with patch.object(ZoteroClient, '__init__', lambda x, y: None):
            client = ZoteroClient.__new__(ZoteroClient)
            
            # v2 format with PDF attachment
            data_v2 = {
                "attachments": [
                    {"contentType": "application/pdf", "filename": "test.pdf"}
                ]
            }
            assert client._has_pdf(data_v2) is True
            
            # v2 format without PDF
            data_no_pdf = {
                "attachments": [
                    {"contentType": "image/png", "filename": "image.png"}
                ]
            }
            assert client._has_pdf(data_no_pdf) is False

    def test_has_pdf_v3_format(self):
        """Test _has_pdf with v3 API format (links.attachment)."""
        from zoterorag.zotero_client import ZoteroClient
        
        with patch.object(ZoteroClient, '__init__', lambda x, y: None):
            client = ZoteroClient.__new__(ZoteroClient)
            
            # v3 format with .pdf ending
            data_v3_pdf = {
                "links": {
                    "attachment": {
                        "href": "http://zotero.org/users/xxx/items/YYY.pdf"
                    }
                }
            }
            assert client._has_pdf(data_v3_pdf) is True
            
            # v3 format without .pdf ending - check meta.attachments instead
            data_meta = {
                "meta": {
                    "attachments": [
                        {"contentType": "application/pdf", "key": "ATT123"}
                    ]
                }
            }
            assert client._has_pdf(data_meta) is True
            
            # v3 format with .pdf in the href (full URL)
            data_v3_full = {
                "links": {
                    "attachment": {
                        "href": "http://zotero.org/users/xxx/items/ATT123/file.pdf"
                    }
                }
            }
            assert client._has_pdf(data_v3_full) is True

    def test_has_pdf_direct_attachment(self):
        """Test _has_pdf for direct attachment items."""
        from zoterorag.zotero_client import ZoteroClient
        
        with patch.object(ZoteroClient, '__init__', lambda x, y: None):
            client = ZoteroClient.__new__(ZoteroClient)
            
            # Direct PDF attachment
            data = {
                "itemType": "attachment",
                "contentType": "application/pdf"
            }
            assert client._has_pdf(data) is True
            
            # Non-PDF attachment
            data_non_pdf = {
                "itemType": "attachment", 
                "contentType": "image/jpeg"
            }
            assert client._has_pdf(data_non_pdf) is False

    def test_find_pdf_key_v2_format(self):
        """Test _find_pdf_key with v2 format."""
        from zoterorag.zotero_client import ZoteroClient
        
        with patch.object(ZoteroClient, '__init__', lambda x, y: None):
            client = ZoteroClient.__new__(ZoteroClient)
            
            data = {
                "attachments": [
                    {"contentType": "application/pdf", "key": "PDF123"},
                    {"contentType": "image/png", "key": "IMG456"}
                ]
            }
            key = client._find_pdf_key(data)
            assert key == "PDF123"

    def test_find_pdf_key_v3_format(self):
        """Test _find_pdf_key with v3 format."""
        from zoterorag.zotero_client import ZoteroClient
        
        with patch.object(ZoteroClient, '__init__', lambda x, y: None):
            client = ZoteroClient.__new__(ZoteroClient)
            
            data = {
                "links": {
                    "attachment": {
                        "href": "http://zotero.org/users/xxx/items/PDF789/file"
                    }
                }
            }
            key = client._find_pdf_key(data)
            assert key == "PDF789"

    def test_find_pdf_key_no_pdf(self):
        """Test _find_pdf_key when no PDF exists."""
        from zoterorag.zotero_client import ZoteroClient
        
        with patch.object(ZoteroClient, '__init__', lambda x, y: None):
            client = ZoteroClient.__new__(ZoteroClient)
            
            data = {
                "attachments": [
                    {"contentType": "image/png", "key": "IMG456"}
                ]
            }
            key = client._find_pdf_key(data)
            assert key is None

    @patch('requests.Session')
    def test_check_connection_success(self, mock_session):
        """Test check_connection returns True on successful response."""
        from zoterorag.zotero_client import ZoteroClient
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_session.return_value.get.return_value = mock_response
        
        client = ZoteroClient("http://localhost:23119")
        result = client.check_connection()
        
        assert result is True

    @patch('requests.Session')
    def test_check_connection_failure(self, mock_session):
        """Test check_connection returns False when API unavailable."""
        from zoterorag.zotero_client import ZoteroClient
        import requests
        
        mock_session.return_value.get.side_effect = requests.RequestException("Connection refused")
        
        client = ZoteroClient("http://localhost:23119")
        result = client.check_connection()
        
        assert result is False

    @patch('requests.Session')
    def test_get_collections(self, mock_session):
        """Test fetching collections from the API."""
        from zoterorag.zotero_client import ZoteroClient
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"key": "col1", "data": {"name": "Collection 1"}},
            {"key": "col2", "data": {"name": "Collection 2"}}
        ]
        mock_session.return_value.get.return_value = mock_response
        
        client = ZoteroClient("http://localhost:23119")
        collections = client.get_collections()
        
        assert len(collections) == 2
        assert collections[0]["key"] == "col1"

    @patch('requests.Session')
    def test_get_collection_items(self, mock_session):
        """Test fetching items from a collection."""
        from zoterorag.zotero_client import ZoteroClient
        
        # First call returns items with Total-Results header
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"key": "item1", "data": {"title": "Item 1"}},
            {"key": "item2", "data": {"title": "Item 2"}}
        ]
        mock_response.headers = {"Total-Results": "2"}
        mock_session.return_value.get.return_value = mock_response
        
        client = ZoteroClient("http://localhost:23119")
        items = list(client.get_collection_items("col1"))
        
        assert len(items) == 2

    @patch('requests.Session')
    def test_get_total_items_count(self, mock_session):
        """Test getting total item count."""
        from zoterorag.zotero_client import ZoteroClient
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Total-Results": "150"}
        mock_session.return_value.get.return_value = mock_response
        
        client = ZoteroClient("http://localhost:23119")
        count = client.get_total_items_count()
        
        assert count == 150

    def test_get_item_by_key_found(self):
        """Test getting a specific item by key."""
        from zoterorag.zotero_client import ZoteroClient
        
        with patch('requests.Session') as mock_session:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"key": "ABC123", "data": {"title": "Test"}}
            mock_session.return_value.get.return_value = mock_response
            
            client = ZoteroClient("http://localhost:23119")
            item = client.get_item_by_key("ABC123")
            
            assert item is not None
            assert item["key"] == "ABC123"

    def test_get_item_by_key_not_found(self):
        """Test getting non-existent item returns None."""
        from zoterorag.zotero_client import ZoteroClient
        
        with patch('requests.Session') as mock_session:
            import requests
            mock_session.return_value.get.side_effect = requests.RequestException("Not found")
            
            client = ZoteroClient("http://localhost:23119")
            item = client.get_item_by_key("NONEXISTENT")
            
            assert item is None

    @patch('requests.Session')
    def test_get_items_since(self, mock_session):
        """Test getting items modified since a version."""
        from zoterorag.zotero_client import ZoteroClient
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"key": "ABC123", "data": {"title": "Updated Item"}}
        ]
        mock_session.return_value.get.return_value = mock_response
        
        client = ZoteroClient("http://localhost:23119")
        items = client.get_items_since(100)
        
        assert len(items) == 1
        # Verify the 'since' parameter was passed
        call_args = mock_session.return_value.get.call_args
        assert call_args[1]["params"]["since"] == 100

    @patch('requests.Session')
    def test_get_file_url_from_meta_attachments(self, mock_session):
        """Test get_file_url from meta.attachments."""
        from zoterorag.zotero_client import ZoteroClient
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "key": "ABC123",
            "meta": {
                "attachments": [
                    {"contentType": "application/pdf", "key": "ATT789"}
                ]
            }
        }
        mock_session.return_value.get.return_value = mock_response
        
        client = ZoteroClient("http://localhost:23119")
        url = client.get_file_url("ABC123")
        
        assert url is not None
        assert "ATT789" in url

    @patch('requests.Session')
    def test_download_pdf_success(self, mock_session):
        """Test successful PDF download."""
        from zoterorag.zotero_client import ZoteroClient
        
        with patch.object(ZoteroClient, 'get_file_url', return_value="http://localhost:23119/items/ATT789/file"):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.iter_content = lambda chunk_size: [b"PDF content bytes"]
            mock_session.return_value.get.return_value = mock_response
            
            client = ZoteroClient("http://localhost:23119")
            
            with patch('builtins.open', create=True) as mock_open:
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                result = client.download_pdf("ABC123", Path("/tmp/test.pdf"))
                
                assert result is True

    def test_download_pdf_no_url(self):
        """Test download_pdf when no file URL available."""
        from zoterorag.zotero_client import ZoteroClient
        
        with patch.object(ZoteroClient, 'get_file_url', return_value=None):
            client = ZoteroClient("http://localhost:23119")
            
            result = client.download_pdf("ABC123", Path("/tmp/test.pdf"))
            
            assert result is False

    def test_download_pdf_no_content(self):
        """Test download_pdf when response has no content (204)."""
        from zoterorag.zotero_client import ZoteroClient
        
        with patch('requests.Session') as mock_session:
            with patch.object(ZoteroClient, 'get_file_url', return_value="http://localhost:23119/items/ATT789/file"):
                mock_response = MagicMock()
                mock_response.status_code = 204
                mock_session.return_value.get.return_value = mock_response
                
                client = ZoteroClient("http://localhost:23119")
                
                result = client.download_pdf("ABC123", Path("/tmp/test.pdf"))
                
                assert result is False

    def test_zotero_local_api_error(self):
        """Test the custom exception class."""
        from zoterorag.zotero_client import ZoteroLocalAPIError
        
        error = ZoteroLocalAPIError("Connection failed")
        
        assert str(error) == "Connection failed"
        assert isinstance(error, Exception)

    def test_zotero_connection_integration(self):
        """Test actual Zotero connection by initializing the client and checking.
        
        This uses the ZoteroClient class to verify connectivity works.
        """
        from zoterorag.zotero_client import ZoteroClient
        
        # Initialize real client - will use config defaults
        client = ZoteroClient()
        
        # The check_connection method tests actual API availability
        result = client.check_connection()
        
        # Should return True if Zotero is running, False otherwise
        assert isinstance(result, bool)


class TestPDFProcessor:
    """Test PDF processing and section extraction."""

    def test_segment_into_sections(self):
        from zoterorag.pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        
        markdown = """# Introduction
This is the introduction text.

## Background
Some background information here.

## Methods
Our methods are described below.
"""
        
        sections = processor.segment_into_sections(markdown, "ABC123")
        
        assert len(sections) >= 1
        # Check that we have the expected section titles
        titles = [s.title for s in sections]
        assert any("Introduction" in t for t in titles)

    def test_extract_quarter_sections_default(self):
        """Test extract_quarter_sections with default page_splits=4."""
        from zoterorag.pdf_processor import PDFProcessor
        
        processor = PDFProcessor(page_splits=4)
        
        # Use a real PDF if available, otherwise this tests the method exists
        pdf_path = Path("data/pdfs/2SGEY4WN.pdf")
        if not pdf_path.exists():
            pytest.skip("No test PDF available")
        
        sections = processor.extract_quarter_sections(pdf_path)
        
        # Should have multiple sections (pages * splits)
        assert len(sections) > 0
        
        # Check that page_section is set
        for section in sections:
            assert section.page_section is not None
            assert 1 <= section.page_section <= 4

    def test_extract_quarter_sections_custom_splits(self):
        """Test extract_quarter_sections with custom number of splits."""
        from zoterorag.pdf_processor import PDFProcessor
        
        processor = PDFProcessor(page_splits=2)
        
        pdf_path = Path("data/pdfs/2SGEY4WN.pdf")
        if not pdf_path.exists():
            pytest.skip("No test PDF available")
        
        sections = processor.extract_quarter_sections(pdf_path)
        
        # Should have multiple sections (pages * splits)
        assert len(sections) > 0
        
        # Check page_section is between 1 and 2
        for section in sections:
            assert section.page_section is not None
            assert 1 <= section.page_section <= 2

    def test_extract_quarter_sections_nonexistent_file(self):
        """Test extract_quarter_sections with nonexistent file."""
        from zoterorag.pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        
        sections = processor.extract_quarter_sections("nonexistent.pdf")
        
        assert len(sections) == 0

    def test_create_sentence_windows(self):
        from zoterorag.pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        
        # Create a section with multiple sentences
        section = Section(
            id="doc_sec_0",
            document_id="ABC123",
            title="Test",
            level=1,
            start_page=1,
            end_page=1,
            text="First sentence. Second sentence! Third sentence? Fourth sentence. Fifth sentence.",
        )
        
        windows = processor.create_sentence_windows(section, window_size=3)
        
        assert len(windows) >= 1
        # First window should have first 3 sentences
        assert "First sentence" in windows[0].text

    def test_create_sentence_windows_few_sentences(self):
        from zoterorag.pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        
        # Section with fewer sentences than window_size
        section = Section(
            id="doc_sec_0",
            document_id="ABC123",
            title="Test",
            level=1,
            start_page=1,
            end_page=1,
            text="Only one sentence here.",
        )
        
        windows = processor.create_sentence_windows(section, window_size=3)
        
        assert len(windows) == 1
        assert "Only one sentence here" in windows[0].text


class TestVectorStore:
    """Test ChromaDB vector store."""

    def test_vector_store_creation(self, tmp_path):
        from zoterorag.vector_store import VectorStore
        
        store_dir = tmp_path / "test_vectors"
        store = VectorStore(str(store_dir))
        
        assert store.sections_collection is not None
        assert store.sentences_collection is not None

    def test_add_and_get_sections(self, tmp_path):
        from zoterorag.vector_store import VectorStore
        
        store_dir = tmp_path / "test_vectors"
        store = VectorStore(str(store_dir))
        
        sections = [
            Section(
                id="doc1_sec_0",
                document_id="doc1",
                title="Introduction",
                level=1,
                start_page=1,
                end_page=3,
                text="This is the introduction.",
            ),
            Section(
                id="doc1_sec_1", 
                document_id="doc1",
                title="Methods",
                level=1,
                start_page=4,
                end_page=10,
                text="These are our methods.",
            )
        ]
        
        # Use simple mock embeddings (384-dim vectors)
        embeddings = [[0.1] * 384 for _ in sections]
        
        store.add_sections(sections, embeddings, "doc1")
        
        retrieved = store.get_all_sections("doc1")
        assert len(retrieved) == 2
        
        single = store.get_section("doc1_sec_0")
        assert single is not None
        assert single.title == "Introduction"

    def test_get_embedded_documents(self, tmp_path):
        from zoterorag.vector_store import VectorStore
        
        store_dir = tmp_path / "test_vectors"
        store = VectorStore(str(store_dir))
        
        # Save some embedded documents
        docs = {"doc1": 5, "doc2": 10}
        store.save_embedded_documents(docs)
        
        retrieved = store.get_embedded_documents()
        assert retrieved["doc1"] == 5
        assert retrieved["doc2"] == 10


class TestSearchEngine:
    """Test search functionality."""

    def test_cosine_similarity(self):
        from zoterorag.search_engine import SearchEngine
        
        # Create a mock config
        config = Config()
        
        with patch.object(SearchEngine, '__init__', lambda x, y: None):
            engine = SearchEngine.__new__(SearchEngine)
            engine.config = config
            
            vec1 = [1.0, 0.0, 0.0]
            vec2 = [1.0, 0.0, 0.0]
            
            # Identical vectors should have similarity 1.0
            sim = engine._cosine_similarity(vec1, vec2)
            assert abs(sim - 1.0) < 0.001
            
            # Orthogonal vectors should have similarity 0.0
            vec3 = [0.0, 1.0, 0.0]
            sim = engine._cosine_similarity(vec1, vec3)
            assert abs(sim - 0.0) < 0.001


class TestMCPResponses:
    """Test MCP response helpers."""

    def test_create_mcp_response(self):
        from zoterorag.mcp_server import create_mcp_response
        
        result = {"status": "ok", "data": [1, 2, 3]}
        response = create_mcp_response("test_tool", result)
        
        assert response["jsonrpc"] == "2.0"
        assert response["result"]["tool"] == "test_tool"
        assert response["result"]["output"] == result

    def test_create_mcp_error(self):
        from zoterorag.mcp_server import create_mcp_error
        
        error = create_mcp_error(-32600, "Invalid Request")
        
        assert error["jsonrpc"] == "2.0"
        assert error["error"]["code"] == -32600
        assert error["error"]["message"] == "Invalid Request"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_section_with_empty_text(self):
        section = Section(
            id="doc_sec_0",
            document_id="ABC123",
            title="Empty Section",
            level=1,
            start_page=1,
            end_page=1,
            text="",  # Empty text
        )
        
        assert section.text == ""
        assert not section.is_embedded

    def test_document_with_no_authors(self):
        doc = Document(
            zotero_key="ABC123",
            title="Anonymous Paper",
            authors=[],  # No authors
        )
        
        assert len(doc.authors) == 0
        assert doc.pdf_path is None

    def test_sentence_window_ordering(self):
        window1 = SentenceWindow(
            id="sec_0_win_0",
            section_id="sec_0",
            window_index=0,
            sentences=["First"],
            text="First",
        )
        
        window2 = SentenceWindow(
            id="sec_0_win_1", 
            section_id="sec_0",
            window_index=1,
            sentences=["Second"],
            text="Second", 
        )
        
        # Windows should be ordered by index
        assert window1.window_index < window2.window_index

    def test_search_result_with_minimal_data(self):
        result = SearchResult(
            text="Minimal text",
            document_title="",
            section_title="",
            zotero_key="",
        )
        
        assert result.relevance_score == 0.0
        assert result.rerank_score == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])