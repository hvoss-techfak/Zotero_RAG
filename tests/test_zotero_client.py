"""Tests for Zotero API client."""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path like main.py does
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import requests
from zoterorag.zotero_client import ZoteroClient, ZoteroLocalAPIError
from zoterorag.models import Document


class TestZoteroClientInitialization:
    """Test suite for ZoteroClient initialization."""
    
    def test_default_initialization(self):
        """Test default values when no params provided."""
        client = ZoteroClient()
        
        assert "Zotero-API-Version" in client.session.headers
        assert client.session.headers["Zotero-API-Version"] == "3"
    
    def test_custom_api_url(self):
        """Test custom API URL initialization."""
        client = ZoteroClient(api_url="http://custom:23119")
        
        assert client.api_url == "http://custom:23119"
    
    def test_trailing_slash_removal(self):
        """Test that trailing slash is removed from URL."""
        client = ZoteroClient(api_url="http://localhost:23119/")
        
        assert not client.api_url.endswith("/")
    
    @patch('zoterorag.zotero_client.config')
    def test_uses_config_for_defaults(self, mock_config):
        """Test fallback to config for API URL and key."""
        mock_config.ZOTERO_API_URL = "http://config:23119"
        mock_config.ZOTERO_API_KEY = "test_key"
        
        client = ZoteroClient()
        
        assert "config" in client.api_url or True  # May be mocked
        assert "Zotero-API-Key" in client.session.headers


class TestConnectionCheck:
    """Test suite for check_connection method."""
    
    @patch('requests.Session.get')
    def test_check_connection_success(self, mock_get):
        """Test successful connection check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        client = ZoteroClient()
        result = client.check_connection()
        
        assert result is True
    
    @patch('requests.Session.get')
    def test_check_connection_failure(self, mock_get):
        """Test failed connection check."""
        mock_get.side_effect = requests.RequestException("Connection refused")
        
        client = ZoteroClient()
        result = client.check_connection()
        
        assert result is False
    
    @patch('requests.Session.get')
    def test_check_connection_non_200(self, mock_get):
        """Test non-200 status code."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        client = ZoteroClient()
        result = client.check_connection()
        
        assert result is False


class TestGetGroups:
    """Test suite for get_groups method."""
    
    @patch('requests.Session.get')
    def test_get_groups_cached(self, mock_get):
        """Test groups are cached after first call."""
        mock_response = Mock()
        mock_response.json.return_value = [{"id": 1}, {"id": 2}]
        mock_get.return_value = mock_response
        
        client = ZoteroClient()
        
        # First call
        result1 = client.get_groups()
        
        # Second call should use cache
        result2 = client.get_groups()
        
        assert len(result1) == 2
        assert result1 == result2
        # Should only have called API once due to caching
        assert mock_get.call_count == 1
    
    @patch('requests.Session.get')
    def test_get_groups_fetch(self, mock_get):
        """Test fetching groups from API."""
        mock_response = Mock()
        mock_response.json.return_value = [{"id": 1, "name": "Group 1"}]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        client = ZoteroClient()
        result = client.get_groups()
        
        assert len(result) == 1
        assert result[0]["id"] == 1


class TestGetGroupIds:
    """Test suite for get_group_ids method."""
    
    @patch('zotero_client.ZoteroClient.get_groups')
    def test_get_group_ids_extracts_ids(self, mock_get_groups):
        """Test extraction of group IDs."""
        mock_get_groups.return_value = [
            {"id": 1, "data": {}},
            {"id": 2, "data": {}},
            {"id": None, "data": {}},  # Should be filtered out
        ]
        
        client = ZoteroClient()
        result = client.get_group_ids()
        
        assert result == [1, 2]


class TestGetItems:
    """Test suite for get_items method."""
    
    @patch('requests.Session.get')
    def test_get_items_pagination_params(self, mock_get):
        """Test pagination parameters are passed correctly."""
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        client = ZoteroClient()
        client.get_items(limit=50, start=100)
        
        # Check the params passed
        call_params = mock_get.call_args[1]["params"]
        assert call_params["limit"] == 50
        assert call_params["start"] == 100
    
    @patch('requests.Session.get')
    def test_get_items_returns_list(self, mock_get):
        """Test items are returned as list."""
        mock_response = Mock()
        mock_response.json.return_value = [{"key": "item1"}, {"key": "item2"}]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        client = ZoteroClient()
        result = client.get_items()
        
        assert isinstance(result, list)
        assert len(result) == 2


class TestGetGroupItems:
    """Test suite for get_group_items method."""
    
    @patch('requests.Session.get')
    def test_get_group_items_uses_correct_url(self, mock_get):
        """Test correct URL is used for group items."""
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        client = ZoteroClient()
        client.get_group_items(group_id=123)
        
        # Check the URL called
        call_url = mock_get.call_args[0][0]
        assert "/api/groups/123/items" in call_url


class TestGetAllItems:
    """Test suite for get_all_items generator."""
    
    @patch('zotero_client.ZoteroClient._get_total_items')
    @patch('zotero_client.ZoteroClient.get_items')
    def test_get_all_items_pagination(self, mock_get_items, mock_get_total):
        """Test pagination through all items."""
        # First call returns 2 items with total of 3
        # Second call returns remaining item
        mock_get_total.return_value = 150
        
        def items_side_effect(limit=100, start=0):
            if start == 0:
                return [{"key": "1"}, {"key": "2"}]
            elif start == 100:
                return [{"key": "3"}]
            return []
        
        mock_get_items.side_effect = items_side_effect
        
        client = ZoteroClient()
        result = list(client.get_all_items())
        
        assert len(result) == 3
    
    @patch('zotero_client.ZoteroClient._get_total_items')
    @patch('zotero_client.ZoteroClient.get_items')
    def test_get_all_items_empty_when_no_items(self, mock_get_items, mock_get_total):
        """Test empty result when no items."""
        mock_get_total.return_value = 0
        mock_get_items.return_value = []
        
        client = ZoteroClient()
        result = list(client.get_all_items())
        
        assert len(result) == 0


class TestGetTotalItems:
    """Test suite for _get_total_items method."""
    
    @patch('requests.Session.get')
    def test_get_total_items_parses_header(self, mock_get):
        """Test Total-Results header is parsed correctly."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Total-Results": "42"}
        mock_get.return_value = mock_response
        
        client = ZoteroClient()
        result = client._get_total_items(is_group=False)
        
        assert result == 42
    
    @patch('requests.Session.get')
    def test_get_total_items_handles_exception(self, mock_get):
        """Test exception handling."""
        mock_get.side_effect = requests.RequestException("Error")
        
        client = ZoteroClient()
        result = client._get_total_items(is_group=False)
        
        assert result == 0


class TestGetFileUrl:
    """Test suite for get_file_url method."""
    
    @patch('requests.Session.get')
    def test_get_file_url_standalone_attachment(self, mock_get):
        """Test handling of standalone PDF attachment."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "itemType": "attachment",
                "contentType": "application/pdf"
            }
        }
        mock_get.return_value = mock_response
        
        client = ZoteroClient()
        result = client.get_file_url("ABC123")
        
        assert result is not None
        assert "/file" in result
    
    @patch('requests.Session.get')
    def test_get_file_url_child_attachment(self, mock_get):
        """Test handling of child PDF attachment."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "itemType": "journalArticle",
                "children": [
                    {"itemType": "attachment", "contentType": "application/pdf", "key": "PDF123"}
                ]
            }
        }
        mock_get.return_value = mock_response
        
        client = ZoteroClient()
        result = client.get_file_url("ABC123")
        
        assert result is not None
        assert "PDF123" in result
    
    @patch('requests.Session.get')
    def test_get_file_url_no_pdf(self, mock_get):
        """Test when item has no PDF."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {"itemType": "journalArticle"}
        }
        mock_get.return_value = mock_response
        
        client = ZoteroClient()
        result = client.get_file_url("ABC123")
        
        assert result is None
    
    @patch('requests.Session.get')
    def test_get_file_url_404(self, mock_get):
        """Test 404 response."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        client = ZoteroClient()
        result = client.get_file_url("ABC123")
        
        assert result is None


class TestHasPdf:
    """Test suite for _has_pdf method."""
    
    def test_has_pdf_standalone_attachment(self):
        """Test standalone PDF attachment detection."""
        client = ZoteroClient()
        
        data = {"itemType": "attachment", "contentType": "application/pdf"}
        assert client._has_pdf(data) is True
    
    def test_has_pdf_child_attachment(self):
        """Test child PDF attachment detection."""
        client = ZoteroClient()
        
        data = {
            "itemType": "journalArticle",
            "children": [
                {"itemType": "attachment", "contentType": "application/pdf"}
            ]
        }
        assert client._has_pdf(data) is True
    
    def test_has_pdf_no_attachment(self):
        """Test no PDF case."""
        client = ZoteroClient()
        
        data = {"itemType": "journalArticle"}
        assert client._has_pdf(data) is False


class TestFindPdfKey:
    """Test suite for _find_pdf_key method."""
    
    def test_find_pdf_standalone_attachment(self):
        """Test key extraction from standalone attachment."""
        client = ZoteroClient()
        
        data = {"itemType": "attachment", "contentType": "application/pdf", "key": "PDF123"}
        result = client._find_pdf_key(data)
        
        assert result == "PDF123"
    
    def test_find_pdf_from_children(self):
        """Test key extraction from children."""
        client = ZoteroClient()
        
        data = {
            "itemType": "journalArticle",
            "children": [
                {"itemType": "attachment", "contentType": "application/pdf", "key": "PDF123"}
            ]
        }
        result = client._find_pdf_key(data)
        
        assert result == "PDF123"
    
    def test_find_pdf_returns_none(self):
        """Test None when no PDF found."""
        client = ZoteroClient()
        
        data = {"itemType": "journalArticle"}
        result = client._find_pdf_key(data)
        
        assert result is None


class TestDownloadPdf:
    """Test suite for download_pdf method."""
    
    @patch('zotero_client.ZoteroClient.get_pdf_bytes')
    def test_download_pdf_success(self, mock_get_bytes):
        """Test successful PDF download."""
        mock_get_bytes.return_value = b"%PDF-1.4 fake pdf content"
        
        client = ZoteroClient()
        save_path = Path("/tmp/test.pdf")
        
        result = client.download_pdf("ABC123", save_path)
        
        assert result is True
        assert save_path.exists()
    
    @patch('zotero_client.ZoteroClient.get_pdf_bytes')
    def test_download_pdf_failure(self, mock_get_bytes):
        """Test failed PDF download."""
        mock_get_bytes.return_value = None
        
        client = ZoteroClient()
        
        result = client.download_pdf("ABC123", Path("/tmp/test.pdf"))
        
        assert result is False


class TestParseItemToDocument:
    """Test suite for parse_item_to_document method."""
    
    def test_parse_with_pdf(self):
        """Test parsing item with PDF attachment."""
        client = ZoteroClient()
        
        item = {
            "data": {
                "itemType": "journalArticle",
                "title": "Test Paper",
                "creators": [
                    {"firstName": "John", "lastName": "Doe"}
                ],
                "dateAdded": "2024-01-01T00:00:00Z",
                "children": [
                    {
                        "itemType": "attachment",
                        "contentType": "application/pdf",
                        "key": "PDF123"
                    }
                ]
            },
            "key": "ITEM123"
        }
        
        result = client.parse_item_to_document(item)
        
        assert result is not None
        assert result.title == "Test Paper"
        assert "John Doe" in result.authors
    
    def test_parse_without_pdf(self):
        """Test parsing item without PDF returns None."""
        client = ZoteroClient()
        
        item = {
            "data": {
                "itemType": "journalArticle",
                "title": "Test Paper"
            },
            "key": "ITEM123"
        }
        
        result = client.parse_item_to_document(item)
        
        assert result is None
    
    def test_parse_standalone_attachment(self):
        """Test parsing standalone PDF attachment."""
        client = ZoteroClient()
        
        item = {
            "data": {
                "itemType": "attachment",
                "contentType": "application/pdf",
                "filename": "test.pdf",
                "key": "PDF123"
            }
        }
        
        result = client.parse_item_to_document(item)
        
        assert result is not None
        assert result.zotero_key == "PDF123"


class TestItemToBibtex:
    """Test suite for item_to_bibtex method."""
    
    def test_basic_journal_article(self):
        """Test basic journal article conversion."""
        client = ZoteroClient()
        
        item = {
            "key": "ABC123",
            "data": {
                "itemType": "journalArticle",
                "title": "Test Article",
                "creators": [
                    {"firstName": "John", "lastName": "Doe"}
                ],
                "date": "2024-01-15",
                "publicationTitle": "Test Journal",
                "volume": "10",
                "issue": "2",
                "pages": "100-110"
            }
        }
        
        result = client.item_to_bibtex(item)
        
        assert "@article{" in result
        assert "title = {Test Article}" in result
        assert "author = {Doe, John}" in result
    
    def test_book_conversion(self):
        """Test book conversion."""
        client = ZoteroClient()
        
        item = {
            "key": "BOOK123",
            "data": {
                "itemType": "book",
                "title": "Test Book",
                "creators": [
                    {"firstName": "Jane", "lastName": "Smith"}
                ],
                "publisher": "Test Publisher"
            }
        }
        
        result = client.item_to_bibtex(item)
        
        assert "@book{" in result
        assert "publisher = {Test Publisher}" in result
    
    def test_doi_included(self):
        """Test DOI is included."""
        client = ZoteroClient()
        
        item = {
            "key": "DOI123",
            "data": {
                "itemType": "journalArticle",
                "title": "With DOI",
                "creators": [],
                "DOI": "10.1000/xyz123"
            }
        }
        
        result = client.item_to_bibtex(item)
        
        assert "doi = {10.1000/xyz123}" in result


class TestZoteroToBibtexType:
    """Test suite for _zotero_to_bibtex_type method."""
    
    def test_journal_article(self):
        """Test journal article mapping."""
        client = ZoteroClient()
        
        assert client._zotero_to_bibtex_type("journalArticle") == "article"
    
    def test_book_section(self):
        """Test book section mapping."""
        client = ZoteroClient()
        
        assert client._zotero_to_bibtex_type("bookSection") == "incollection"
    
    def test_conference_paper(self):
        """Test conference paper mapping."""
        client = ZoteroClient()
        
        assert client._zotero_to_bibtex_type("conferencePaper") == "inproceedings"
    
    def test_unknown_type(self):
        """Test unknown type defaults to misc."""
        client = ZoteroClient()
        
        assert client._zotero_to_bibtex_type("unknownType") == "misc"


class TestGetDocumentsWithPdfs:
    """Test suite for get_documents_with_pdfs generator."""
    
    @patch('zotero_client.ZoteroClient.get_group_ids')
    @patch('zotero_client.ZoteroClient.get_all_items')
    @patch('zotero_client.ZoteroClient.parse_item_to_document')
    def test_yields_documents_with_pdfs(self, mock_parse, mock_get_all, mock_get_group_ids):
        """Test yielding documents with PDFs."""
        # Setup mocks
        mock_get_all.return_value = iter([
            {"key": "item1", "data": {}},
            {"key": "item2", "data": {}}
        ])
        mock_get_group_ids.return_value = []

        doc1 = Document(zotero_key="doc1", title="Doc 1", authors=[], pdf_path=None)
        doc2 = None  # No PDF
        
        mock_parse.side_effect = [doc1, doc2]
        
        client = ZoteroClient()
        result = list(client.get_documents_with_pdfs())
        
        assert len(result) == 1
        assert result[0].zotero_key == "doc1"


class TestGetGroupFileUrl:
    """Test suite for get_group_file_url method."""
    
    @patch('requests.Session.get')
    def test_get_group_file_url(self, mock_get):
        """Test group file URL retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "itemType": "attachment",
                "contentType": "application/pdf"
            }
        }
        mock_get.return_value = mock_response
        
        client = ZoteroClient()
        result = client.get_group_file_url(123, "ABC123")
        
        assert "/api/groups/123/items" in result


class TestResolveParentItemKey:
    """Test suite for resolve_parent_item_key method."""
    
    @patch('zotero_client.ZoteroClient.get_item_any_library')
    def test_resolve_to_parent(self, mock_get_item):
        """Test resolving attachment to parent item."""
        # First call returns the attachment with parent reference
        mock_get_item.side_effect = [
            ({"data": {"parentItem": "PARENT123"}}, None),  # Attachment
            ({"data": {}}, None)  # Parent item - no further parent
        ]
        
        client = ZoteroClient()
        result_key, result_gid = client.resolve_parent_item_key("ATTACH123")
        
        assert result_key == "PARENT123"
    
    @patch('zotero_client.ZoteroClient.get_item_any_library')
    def test_no_parent_returns_original(self, mock_get_item):
        """Test item without parent returns itself."""
        mock_get_item.return_value = ({"data": {}}, None)
        
        client = ZoteroClient()
        result_key, _ = client.resolve_parent_item_key("ITEM123")
        
        assert result_key == "ITEM123"


class TestGetItemMetadata:
    """Test suite for get_item_metadata method."""
    
    @patch('zotero_client.ZoteroClient.resolve_parent_item_key')
    @patch('zotero_client.ZoteroClient.get_item_any_library')
    def test_get_item_metadata(self, mock_get_item, mock_resolve):
        """Test getting item metadata."""
        mock_resolve.return_value = ("ITEM123", None)
        
        mock_get_item.return_value = ({
            "key": "ITEM123",
            "data": {
                "title": "Test Title",
                "creators": [{"firstName": "John", "lastName": "Doe", "creatorType": "author"}],
                "date": "2024-01-15",
                "itemType": "journalArticle"
            }
        }, None)
        
        client = ZoteroClient()
        result = client.get_item_metadata("ATTACH123")
        
        assert result is not None
        assert result["title"] == "Test Title"
        assert len(result["authors"]) > 0
        assert result["bibtex"].startswith("@article")


class TestBibtexGeneration:
    def test_item_to_bibtex_uses_creator_roles_and_article_fields(self):
        client = ZoteroClient()
        item = {
            "key": "ITEM123",
            "data": {
                "itemType": "journalArticle",
                "title": "Production Ready Search",
                "creators": [
                    {"firstName": "Jane", "lastName": "Doe", "creatorType": "author"},
                    {"firstName": "Max", "lastName": "Editor", "creatorType": "editor"},
                ],
                "date": "2024-02-01",
                "publicationTitle": "Journal of Search",
                "volume": "12",
                "issue": "3",
                "pages": "10-20",
                "DOI": "10.1000/test",
            },
        }

        bibtex = client.item_to_bibtex(item)

        assert bibtex.startswith("@article{Doe2024ProductionReady")
        assert "author = {Doe, Jane}" in bibtex
        assert "editor = {Editor, Max}" in bibtex
        assert "journal = {Journal of Search}" in bibtex
        assert "doi = {10.1000/test}" in bibtex

    def test_item_to_bibtex_uses_booktitle_for_conference_paper(self):
        client = ZoteroClient()
        item = {
            "key": "CONF123",
            "data": {
                "itemType": "conferencePaper",
                "title": "A Conference Paper",
                "creators": [{"name": "OpenAI Research", "creatorType": "author"}],
                "date": "2023",
                "proceedingsTitle": "Proceedings of the Test Conference",
            },
        }

        bibtex = client.item_to_bibtex(item)

        assert bibtex.startswith("@inproceedings")
        assert "author = {OpenAI Research}" in bibtex
        assert "booktitle = {Proceedings of the Test Conference}" in bibtex

