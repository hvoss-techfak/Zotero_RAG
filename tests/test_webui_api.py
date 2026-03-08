"""API tests for web UI search and embedding endpoints."""

import sys
import os

# Add the project root to sys.path so we can import webui
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock


class TestWebUIAPI:
    """Test cases for Flask web UI endpoints."""

    @pytest.fixture
    def client(self):
        try:
            from webui.app import app as flask_app

            flask_app.config["TESTING"] = True
            with flask_app.test_client() as client:
                yield client
        except ImportError as e:
            pytest.skip(f"Flask not available: {e}")

    def test_search_endpoint_returns_json(self, client):
        with patch("webui.app.get_server") as mock_get_server:
            mock_server = MagicMock()

            async def mock_search(*args, **kwargs):
                return []

            mock_server.search_documents = mock_search
            mock_get_server.return_value = mock_server

            response = client.post("/api/search", json={"query": "test"})

            assert response.status_code == 200
            data = response.get_json()
            assert "results" in data
            assert "count" in data
            assert "query" in data

    def test_search_endpoint_with_query(self, client):
        mock_results = [
            {
                "text": "Sample result text about machine learning",
                "document_title": "Test Document",
                "authors": ["Author One", "Author Two"],
                "date": "2024-01-15",
                "bibtex": "@article{doc, title={Test Document}}",
                "cited_bibtex": ["@article{test, author={One}}"],
            }
        ]

        with patch("webui.app.get_server") as mock_get_server:
            mock_server = MagicMock()

            async def mock_search(
                query,
                top_sentences=10,
                min_relevance=0.75,
                citation_return_mode="both",
                require_cited_bibtex=False,
            ):
                return mock_results

            mock_server.search_documents = mock_search
            mock_get_server.return_value = mock_server

            response = client.post(
                "/api/search",
                json={
                    "query": "machine learning",
                    "top_sentences": 5,
                    "min_relevance": 0.3,
                },
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["results"] == mock_results
            assert data["count"] == 1
            assert data["query"] == "machine learning"

    def test_search_endpoint_empty_query(self, client):
        response = client.post("/api/search", json={"query": ""})
        assert response.status_code == 400

    def test_search_endpoint_missing_query(self, client):
        response = client.post("/api/search", json={})
        assert response.status_code == 400

    def test_search_endpoint_handles_exception(self, client):
        with patch("webui.app.get_server") as mock_get_server:
            mock_server = MagicMock()

            async def mock_search(*args, **kwargs):
                raise Exception("Search failed")

            mock_server.search_documents = mock_search
            mock_get_server.return_value = mock_server

            response = client.post("/api/search", json={"query": "test"})
            assert response.status_code == 500

    def test_status_endpoint_returns_json(self, client):
        with patch("webui.app.get_server") as mock_get_server:
            mock_server = MagicMock()

            async def mock_status():
                return {
                    "state": "running",
                    "processed_documents": 2,
                    "total_documents": 5,
                }

            mock_server.get_embedding_status = mock_status
            mock_get_server.return_value = mock_server

            response = client.get("/api/status")

            assert response.status_code == 200
            data = response.get_json()
            assert data["state"] == "running"
            assert data["processed_documents"] == 2

    def test_embed_trigger_endpoint_returns_started(self, client):
        with patch("webui.app.get_server") as mock_get_server:
            mock_server = MagicMock()

            async def mock_embed():
                return {"status": "started", "trigger": "manual"}

            mock_server.embed_new_documents_now = mock_embed
            mock_get_server.return_value = mock_server

            response = client.post("/api/embed", json={})

            assert response.status_code == 202
            data = response.get_json()
            assert data["status"] == "started"


class TestWebUIAvailability:
    """Test that web UI can be imported and created."""

    def test_webui_import(self):
        try:
            from webui import app

            assert app is not None
        except ImportError as e:
            pytest.fail(f"Failed to import webui.app: {e}")

    def test_flask_app_exists(self):
        try:
            from webui.app import app

            assert app is not None
            assert hasattr(app, "view_functions")
        except ImportError as e:
            pytest.skip(f"Flask not available: {e}")
