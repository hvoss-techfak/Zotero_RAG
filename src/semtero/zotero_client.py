"""Zotero API client for connecting to a running Zotero instance.

This client connects to the local Zotero API (localhost:23119 by default).
It supports both user library and group libraries.
"""

import logging
import re
import unicodedata
from contextlib import closing
import requests
from pathlib import Path
from typing import Generator

from .config import config
from .models import Document

logger = logging.getLogger(__name__)


class ZoteroLocalAPIError(Exception):
    """Raised when the local Zotero API is not available."""

    pass


class ZoteroClient:
    """Client for interacting with the local Zotero API.

    Supports accessing items from both user library and group libraries.
    """

    # Local API version
    API_VERSION = 3

    def __init__(self, api_url: str | None = None, api_key: str | None = None):
        self.api_url = (api_url or config.ZOTERO_API_URL).rstrip("/")
        self.session = requests.Session()

        # Set required headers for local API
        self.session.headers.update(
            {
                "Zotero-API-Version": str(self.API_VERSION),
                "Zotero-Allowed-Request": "1",
            }
        )

        # Use provided key, then config key, otherwise empty for local connector
        key = api_key or config.ZOTERO_API_KEY
        if key:
            self.session.headers.update({"Zotero-API-Key": key})

        # Cache for groups
        self._groups: list[dict] | None = None

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _get_user_items_url(self) -> str:
        """Get the URL for user items endpoint."""
        return f"{self.api_url}/api/users/0/items"

    def _get_group_items_url(self, group_id: int) -> str:
        """Get the URL for group items endpoint."""
        return f"{self.api_url}/api/groups/{group_id}/items"

    def check_connection(self) -> bool:
        """Check if the local Zotero API is available."""
        try:
            response = self.session.get(f"{self.api_url}/api/")
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_groups(self) -> list[dict]:
        """Fetch all groups for the current user.

        Returns:
            List of group dictionaries with 'id' and 'data' keys
        """
        if self._groups is not None:
            return self._groups

        url = f"{self.api_url}/api/users/0/groups"
        response = self.session.get(url)
        response.raise_for_status()
        self._groups = response.json()
        return self._groups

    def get_group_ids(self) -> list[int]:
        """Get all group IDs for the current user.

        Returns:
            List of group IDs
        """
        groups = self.get_groups()
        return [g.get("id") for g in groups if g.get("id")]

    def get_items(self, limit: int = 100, start: int = 0) -> list[dict]:
        """Fetch items from the user's Zotero library.

        Args:
            limit: Maximum number of items to return
            start: Starting offset for pagination

        Returns:
            List of item dictionaries
        """
        url = self._get_user_items_url()
        params = {"limit": limit, "start": start}
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_group_items(
        self, group_id: int, limit: int = 100, start: int = 0
    ) -> list[dict]:
        """Fetch items from a specific group's library.

        Args:
            group_id: The group ID
            limit: Maximum number of items to return
            start: Starting offset for pagination

        Returns:
            List of item dictionaries
        """
        url = self._get_group_items_url(group_id)
        params = {"limit": limit, "start": start}
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def _get_total_items(
        self, is_group: bool = False, group_id: int | None = None
    ) -> int:
        """Get total item count for pagination.

        Args:
            is_group: Whether to query group library
            group_id: The group ID (required if is_group is True)

        Returns:
            Total number of items or 0 if unknown
        """
        try:
            if is_group and group_id:
                url = self._get_group_items_url(group_id)
            else:
                url = self._get_user_items_url()
            response = self.session.get(url, params={"limit": 1})
            if response.status_code == 200:
                return int(response.headers.get("Total-Results", 0))
        except (requests.RequestException, ValueError):
            pass
        return 0

    def get_all_items(self) -> Generator[dict, None, None]:
        """Fetch all items from the user's library using pagination."""
        total = self._get_total_items(is_group=False)
        start = 0
        limit = 100

        while True:
            if total > 0 and start >= total:
                break
            items = self.get_items(limit=limit, start=start)
            if not items:
                break
            yield from items
            start += limit

    def get_all_group_items(self) -> Generator[dict, None, None]:
        """Fetch all items from all group libraries using pagination."""
        for group_id in self.get_group_ids():
            total = self._get_total_items(is_group=True, group_id=group_id)
            start = 0
            limit = 100

            while True:
                if total > 0 and start >= total:
                    break
                items = self.get_group_items(group_id, limit=limit, start=start)
                if not items:
                    break
                yield from items
                start += limit

    def get_all_items_from_all_libraries(self) -> Generator[dict, None, None]:
        """Fetch all items from user library and all group libraries.

        Yields:
            Item dictionaries from user library first, then each group
        """
        # First yield user's own items
        yield from self.get_all_items()

        # Then yield items from each group
        yield from self.get_all_group_items()

    def get_item(self, item_key: str) -> dict:
        """Fetch a specific item by key from user library.

        Args:
            item_key: The item key

        Returns:
            Item dictionary
        """
        url = f"{self.api_url}/api/users/0/items/{item_key}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_group_item(self, group_id: int, item_key: str) -> dict:
        """Fetch a specific item by key from a group's library.

        Args:
            group_id: The group ID
            item_key: The item key

        Returns:
            Item dictionary
        """
        url = f"{self.api_url}/api/groups/{group_id}/items/{item_key}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_collections(self) -> list[dict]:
        """Fetch all collections from the user's library.

        Returns:
            List of collection dictionaries
        """
        url = f"{self.api_url}/api/users/0/collections"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_group_collections(self, group_id: int) -> list[dict]:
        """Fetch all collections from a group's library.

        Args:
            group_id: The group ID

        Returns:
            List of collection dictionaries
        """
        url = f"{self.api_url}/api/groups/{group_id}/collections"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_collection_items(self, collection_key: str) -> Generator[dict, None, None]:
        """Fetch all items in a specific collection from user library.

        Args:
            collection_key: The key of the collection

        Yields:
            Items belonging to the collection
        """
        url = f"{self.api_url}/api/users/0/collections/{collection_key}/items"
        start = 0
        limit = 100

        while True:
            params = {"limit": limit, "start": start}
            response = self.session.get(url, params=params)
            response.raise_for_status()
            items = response.json()

            if not items:
                break

            yield from items

            # Check if there are more items (local API returns totalResults in header)
            total = int(response.headers.get("Total-Results", 0))
            start += limit
            if start >= total:
                break

    def get_group_collection_items(
        self, group_id: int, collection_key: str
    ) -> Generator[dict, None, None]:
        """Fetch all items in a specific collection from a group's library.

        Args:
            group_id: The group ID
            collection_key: The key of the collection

        Yields:
            Items belonging to the collection
        """
        url = f"{self.api_url}/api/groups/{group_id}/collections/{collection_key}/items"
        start = 0
        limit = 100

        while True:
            params = {"limit": limit, "start": start}
            response = self.session.get(url, params=params)
            response.raise_for_status()
            items = response.json()

            if not items:
                break

            yield from items

            # Check if there are more items (local API returns totalResults in header)
            total = int(response.headers.get("Total-Results", 0))
            start += limit
            if start >= total:
                break

    def get_file_url(self, item_key: str) -> str | None:
        """Get the download URL for an item's PDF file from user library.

        Args:
            item_key: The item key

        Returns:
            File URL or None if not available
        """
        try:
            # Get the item to find attachment info
            url = f"{self.api_url}/api/users/0/items/{item_key}"
            response = self.session.get(url)
            if response.status_code != 200:
                return None

            item = response.json()

            # Check for attachments in data.links (v3 API format)
            data = item.get("data", {})

            # Handle standalone PDF attachment - this IS the file itself
            if (
                data.get("itemType") == "attachment"
                and data.get("contentType") == "application/pdf"
            ):
                return f"{self.api_url}/api/users/0/items/{item_key}/file"

            links = data.get("links", {})
            attachment_link = links.get("attachment", {})
            href = attachment_link.get("href", "")

            if href and ".pdf" in href.lower():
                # Return the file endpoint URL
                key_part = (
                    href.split("/items/")[-1].split("/")[0]
                    if "/items/" in href
                    else None
                )
                if key_part:
                    return f"{self.api_url}/api/users/0/items/{key_part}/file"

            # Check for children attachments (embedded files)
            children = data.get("children", [])
            for child in children:
                if child.get("itemType") == "attachment":
                    content_type = child.get("contentType", "")
                    if (
                        content_type == "application/pdf"
                        or ".pdf" in str(child.get("filename", "")).lower()
                    ):
                        return f"{self.api_url}/api/users/0/items/{child['key']}/file"

            # Check v2 format: data.attachments array
            attachments = data.get("attachments", [])
            for att in attachments:
                content_type = att.get("contentType", "")
                filename = att.get("filename", "") or att.get("path", "")
                if content_type == "application/pdf" or (
                    filename and ".pdf" in filename.lower()
                ):
                    return f"{self.api_url}/api/users/0/items/{att['key']}/file"

            # Check v3 format: data.meta.attachments or item.meta.attachments
            meta = data.get("meta") or item.get("meta", {})
            if meta:
                meta_attachments = meta.get("attachments", [])
                for att in meta_attachments:
                    if att.get("contentType") == "application/pdf":
                        return f"{self.api_url}/api/users/0/items/{att['key']}/file"

        except requests.RequestException:
            pass

        return None

    def get_group_file_url(self, group_id: int, item_key: str) -> str | None:
        """Get the download URL for an item's PDF file from a group's library.

        Args:
            group_id: The group ID
            item_key: The item key

        Returns:
            File URL or None if not available
        """
        try:
            url = f"{self.api_url}/api/groups/{group_id}/items/{item_key}"
            response = self.session.get(url)
            if response.status_code != 200:
                return None

            item = response.json()

            # Check for attachments in data.links (v3 API format)
            data = item.get("data", {})

            # Handle standalone PDF attachment - this IS the file itself
            if (
                data.get("itemType") == "attachment"
                and data.get("contentType") == "application/pdf"
            ):
                return f"{self.api_url}/api/groups/{group_id}/items/{item_key}/file"

            links = data.get("links", {})
            attachment_link = links.get("attachment", {})
            href = attachment_link.get("href", "")

            if href and ".pdf" in href.lower():
                key_part = (
                    href.split("/items/")[-1].split("/")[0]
                    if "/items/" in href
                    else None
                )
                if key_part:
                    return f"{self.api_url}/api/groups/{group_id}/items/{key_part}/file"

            # Check for children attachments (PDFs attached to regular items)
            children = data.get("children", [])
            for child in children:
                if child.get("itemType") == "attachment":
                    content_type = child.get("contentType", "")
                    if (
                        content_type == "application/pdf"
                        or ".pdf" in str(child.get("filename", "")).lower()
                    ):
                        return f"{self.api_url}/api/groups/{group_id}/items/{child['key']}/file"

            # Check v2 format: data.attachments array
            attachments = data.get("attachments", [])
            for att in attachments:
                content_type = att.get("contentType", "")
                filename = att.get("filename", "") or att.get("path", "")
                if content_type == "application/pdf" or (
                    filename and ".pdf" in filename.lower()
                ):
                    return (
                        f"{self.api_url}/api/groups/{group_id}/items/{att['key']}/file"
                    )

            # Check v3 format: data.meta.attachments or item.meta.attachments
            meta = data.get("meta") or item.get("meta", {})
            if meta:
                meta_attachments = meta.get("attachments", [])
                for att in meta_attachments:
                    if att.get("contentType") == "application/pdf":
                        return f"{self.api_url}/api/groups/{group_id}/items/{att['key']}/file"

        except requests.RequestException:
            pass

        return None

    def download_pdf(self, item_key: str, save_path: Path) -> bool:
        """Download the PDF file for an item from user library.

        Args:
            item_key: The Zotero item key
            save_path: Where to save the PDF

        Returns:
            True if successful, False otherwise
        """
        pdf_bytes = self.get_pdf_bytes(item_key)
        if pdf_bytes is None:
            return False

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(pdf_bytes)
            return True
        except OSError:
            return False

    def get_pdf_bytes(self, item_key: str) -> bytes | None:
        """Get PDF content as bytes for an item from user library.

        Args:
            item_key: The Zotero item key

        Returns:
            PDF content as bytes, or None if not available
        """
        url = self.get_file_url(item_key)
        if not url:
            return None

        try:
            # Follow redirect manually since we need the actual file URL.
            # The streamed response must be closed explicitly so repeated
            # background embedding passes don't leak sockets/file descriptors.
            with closing(
                self.session.get(url, stream=True, allow_redirects=False)
            ) as response:
                if response.status_code == 302:
                    redirect_url = response.headers.get("Location")
                    if redirect_url and redirect_url.startswith("file://"):
                        source_path = redirect_url.replace("file://", "")
                        data = self._read_local_pdf(source_path)
                        if not data:
                            return None
                        return data

                if response.status_code == 204:
                    return None

                response.raise_for_status()

                data = response.content
                if not data:
                    return None
                return data

        except requests.RequestException:
            return None

    def _read_local_pdf(self, source_path: str) -> bytes | None:
        """Read a local PDF file.

        Args:
            source_path: The absolute path from Zotero's file:// URL (URL-encoded)

        Returns:
            File content as bytes, or None if not available
        """
        from urllib.parse import unquote

        # Decode URL-encoded path
        decoded_path = unquote(source_path)
        source = Path(decoded_path)

        if not source.exists():
            logger.warning(f"Local PDF not found: {decoded_path}")
            return None

        try:
            return source.read_bytes()
        except OSError as e:
            logger.warning(f"Failed to read PDF from {source}: {e}")
            return None

    def _has_pdf(self, data: dict) -> bool:
        """Check if the item has a PDF attachment."""
        # Check if it's an attachment with PDF content type (standalone attachment)
        if data.get("itemType") == "attachment":
            return data.get("contentType") == "application/pdf"

        # Check for children attachments (embedded files in regular items)
        children = data.get("children", [])
        for child in children:
            if child.get("itemType") == "attachment":
                content_type = child.get("contentType", "")
                if (
                    content_type == "application/pdf"
                    or ".pdf" in str(child.get("filename", "")).lower()
                ):
                    return True

        # Check links.attachment (v3 API format)
        links = data.get("links", {})
        attachment_link = links.get("attachment", {})
        href = attachment_link.get("href", "")
        if href and ".pdf" in href.lower():
            return True

        # Check v2 format: data.attachments array
        attachments = data.get("attachments", [])
        for att in attachments:
            content_type = att.get("contentType", "")
            filename = att.get("filename", "") or att.get("path", "")
            if content_type == "application/pdf" or (
                filename and ".pdf" in filename.lower()
            ):
                return True

        # Check v3 format: data.meta.attachments array
        meta = data.get("meta", {})
        meta_attachments = meta.get("attachments", [])
        for att in meta_attachments:
            if att.get("contentType") == "application/pdf":
                return True

        return False

    def _find_pdf_key(self, data: dict) -> str | None:
        """Find the key of a PDF attachment for this item."""
        # Check if THIS item is a standalone PDF attachment
        if (
            data.get("itemType") == "attachment"
            and data.get("contentType") == "application/pdf"
        ):
            return data.get("key")

        # Check children attachments (for regular items with PDF children)
        children = data.get("children", [])
        for child in children:
            if child.get("itemType") == "attachment":
                content_type = child.get("contentType", "")
                filename = child.get("filename", "")
                if content_type == "application/pdf" or str(filename).lower().endswith(
                    ".pdf"
                ):
                    return child.get("key")

        # Check links.attachment (v3 format)
        links = data.get("links", {})
        attachment_link = links.get("attachment", {})
        href = attachment_link.get("href", "")

        # Extract key from URL like http://zotero.org/users/xxx/items/YYY/file
        if "items/" in href:
            parts = href.split("items/")
            if len(parts) > 1:
                key_part = parts[1].split("/")[0]
                return key_part

        # Check v2 format: data.attachments array
        attachments = data.get("attachments", [])
        for att in attachments:
            content_type = att.get("contentType", "")
            filename = att.get("filename", "") or att.get("path", "")
            if content_type == "application/pdf" or (
                filename and ".pdf" in filename.lower()
            ):
                return att.get("key")

        # Check v3 format: data.meta.attachments array
        meta = data.get("meta", {})
        meta_attachments = meta.get("attachments", [])
        for att in meta_attachments:
            if att.get("contentType") == "application/pdf":
                return att.get("key")

        return None

    def get_documents_with_pdfs(self) -> Generator[Document, None, None]:
        """Get all documents that have PDF files from user and group libraries."""
        # First process user's own library (group_id=None)
        for item in self.get_all_items():
            doc = self.parse_item_to_document(item)
            if doc:
                yield doc

        # Then process each group's library using pagination
        for group_id in self.get_group_ids():
            total = self._get_total_items(is_group=True, group_id=group_id)
            start = 0
            limit = 100

            while True:
                if total > 0 and start >= total:
                    break
                items = self.get_group_items(group_id, limit=limit, start=start)
                if not items:
                    break
                for item in items:
                    try:
                        doc = self.parse_item_to_document(item, group_id=group_id)
                    except StopIteration:
                        return
                    if doc:
                        yield doc
                start += limit

    def get_group_pdf_bytes(self, group_id: int, item_key: str) -> bytes | None:
        """Get PDF content as bytes for an item from a group's library.

        Args:
            group_id: The group ID
            item_key: The Zotero item key

        Returns:
            PDF content as bytes, or None if not available
        """
        url = self.get_group_file_url(group_id, item_key)
        if not url:
            return None

        try:
            with closing(
                self.session.get(url, stream=True, allow_redirects=False)
            ) as response:
                if response.status_code == 302:
                    redirect_url = response.headers.get("Location")
                    if redirect_url and redirect_url.startswith("file://"):
                        source_path = redirect_url.replace("file://", "")
                        data = self._read_local_pdf(source_path)
                        if not data:
                            return None
                        return data

                if response.status_code == 204:
                    return None

                response.raise_for_status()

                data = response.content
                if not data:
                    return None
                return data

        except requests.RequestException:
            return None

    def download_pdf_for_doc(self, document: Document) -> bool:
        """Download PDF for a document. Supports both user and group libraries.

        Args:
            document: The Document to download

        Returns:
            True if successful, False otherwise
        """
        pdf_key = document.zotero_key

        # Check if already downloaded
        if document.pdf_path and document.pdf_path.exists():
            return True

        try:
            # Use group-specific or user method
            if document.group_id is not None:
                pdf_bytes = self.get_group_pdf_bytes(document.group_id, pdf_key)
            else:
                pdf_bytes = self.get_pdf_bytes(pdf_key)

            if pdf_bytes is None:
                return False

            # Save to pdf_path
            save_path = document.pdf_path or config.PDF_CACHE_PATH / f"{pdf_key}.pdf"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "wb") as f:
                f.write(pdf_bytes)

            # Update document's pdf_path
            document.pdf_path = save_path
            return True

        except (requests.RequestException, OSError) as e:
            logger.warning(f"Failed to download PDF {pdf_key}: {e}")
            return False

    def _copy_local_pdf(self, source_path: str, document: Document) -> bool:
        """Copy a local file:// URL path to the cache directory.

        Args:
            source_path: The absolute path from Zotero's file:// URL (URL-encoded)
            document: The Document to update with pdf_path

        Returns:
            True if successful, False otherwise
        """
        import shutil
        from urllib.parse import unquote

        # Decode URL-encoded path (Zotero returns e.g. %20 for space, %E2%80%93 for en-dash)
        decoded_path = unquote(source_path)

        source = Path(decoded_path)

        # Check if the file exists at the given path
        if not source.exists():
            logger.warning(f"Local PDF not found: {decoded_path}")
            return False

        try:
            save_path = (
                document.pdf_path
                or config.PDF_CACHE_PATH / f"{document.zotero_key}.pdf"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy the file to our cache
            shutil.copy2(source, save_path)

            # Update document's pdf_path
            document.pdf_path = save_path
            logger.info(f"Copied PDF from {source} to {save_path}")
            return True

        except OSError as e:
            logger.warning(f"Failed to copy PDF from {source}: {e}")
            return False

    def parse_item_to_document(
        self, item: dict, group_id: int | None = None
    ) -> Document | None:
        """Parse a Zotero API item into a Document model."""
        data = item.get("data", {})

        # Only process items with PDFs
        if not self._has_pdf(data):
            return None

        # Get title - for standalone attachments use filename as fallback
        title = data.get("title", "")
        if not title or title == "Full Text":
            title = data.get("filename", "Untitled")

        authors = [
            a.get("firstName", "") + " " + a.get("lastName", "").strip()
            for a in data.get("creators", [])
        ]

        # Get PDF key and path - use the item's own key if it's an attachment
        pdf_key = self._find_pdf_key(data)
        pdf_path = None
        if pdf_key:
            pdf_path = config.PDF_CACHE_PATH / f"{pdf_key}.pdf"

        # For standalone attachments, also store parent item key for metadata lookup
        parent_item_key = (
            data.get("parentItem") if data.get("itemType") == "attachment" else None
        )

        return Document(
            zotero_key=pdf_key or item.get("key", ""),
            title=title,
            authors=authors,
            pdf_path=pdf_path,
            added_date=data.get("dateAdded", ""),
            modified_date=data.get("dateModified", ""),
            parent_item_key=parent_item_key,  # Track parent for metadata
            group_id=group_id,  # Which library it came from
        )

    def _get_item_url(self, item_key: str, group_id: int | None = None) -> str:
        """Build an item URL for user or group library."""
        if group_id is None:
            return f"{self.api_url}/api/users/0/items/{item_key}"
        return f"{self.api_url}/api/groups/{group_id}/items/{item_key}"

    def _get_file_url(self, item_key: str, group_id: int | None = None) -> str:
        """Build a file URL for user or group library."""
        if group_id is None:
            return f"{self.api_url}/api/users/0/items/{item_key}/file"
        return f"{self.api_url}/api/groups/{group_id}/items/{item_key}/file"

    def get_item_any_library(
        self, item_key: str, group_id: int | None = None
    ) -> tuple[dict | None, int | None]:
        """Fetch an item, optionally trying user and then all known groups.

        Returns (item_json, resolved_group_id). resolved_group_id None means user library.
        """
        # Try the hinted library first
        try_order: list[int | None]
        if group_id is None:
            try_order = [None] + self.get_group_ids()
        else:
            try_order = [group_id]

        for gid in try_order:
            try:
                url = self._get_item_url(item_key, gid)
                resp = self.session.get(url)
                if resp.status_code == 200:
                    return resp.json(), gid
            except requests.RequestException:
                continue

        return None, None

    def resolve_parent_item_key(
        self,
        item_key: str,
        group_id: int | None = None,
        max_depth: int = 10,
    ) -> tuple[str, int | None]:
        """Resolve an attachment/note/etc. up to the bibliographic parent item.

        Returns (resolved_item_key, resolved_group_id).
        """
        visited: set[tuple[int | None, str]] = set()
        current_key = item_key
        current_gid = group_id

        for _ in range(max_depth):
            state = (current_gid, current_key)
            if state in visited:
                break
            visited.add(state)

            item, resolved_gid = self.get_item_any_library(current_key, current_gid)
            if not item:
                # Can't fetch current item; stop and return what we have
                return current_key, current_gid

            current_gid = resolved_gid
            data = item.get("data", {})
            parent_key = data.get("parentItem")

            # If there's no parent, current_key is the best we can do
            if not parent_key:
                return current_key, current_gid

            current_key = parent_key

        return current_key, current_gid

    def get_item_metadata(
        self, item_key: str, group_id: int | None = None
    ) -> dict | None:
        """Get full metadata for an item including BibTeX and file info.

        Traverses the parent chain to find bibliographic metadata. Works for both
        user and group libraries.
        """
        # First, resolve to the bibliographic parent (if any)
        resolved_key, resolved_gid = self.resolve_parent_item_key(
            item_key, group_id=group_id
        )

        # Fetch the resolved item
        item, resolved_gid = self.get_item_any_library(resolved_key, resolved_gid)
        if not item:
            return None

        data = item.get("data", {})
        authors = [
            (a.get("firstName", "") + " " + a.get("lastName", "")).strip()
            for a in data.get("creators", [])
            if (a.get("firstName") or a.get("lastName"))
        ]
        date = data.get("date", "")
        title = data.get("title", "")
        item_type = data.get("itemType", "")

        # File path: keep pointing at the originally-known attachment key
        file_path = self._get_file_url(item_key, resolved_gid)

        return {
            "bibtex": self.item_to_bibtex(item),
            "file_path": file_path,
            "title": title,
            "authors": authors,
            "date": date,
            "item_type": item_type,
        }

    def get_total_items_count(self) -> int:
        """Get the total number of items in the user's library.

        Returns:
            Total count or -1 if unknown
        """
        try:
            url = self._get_user_items_url()
            response = self.session.get(url, params={"limit": 1})
            if response.status_code == 200:
                # Local API returns Total-Results header
                return int(response.headers.get("Total-Results", 0))
        except requests.RequestException:
            pass
        return -1

    def get_item_by_key(self, item_key: str) -> dict | None:
        """Get a single item by its key from user library.

        Args:
            item_key: The Zotero item key

        Returns:
            Item data or None if not found
        """
        try:
            url = f"{self.api_url}/api/users/0/items/{item_key}"
            response = self.session.get(url)
            if response.status_code == 200:
                return response.json()
        except requests.RequestException:
            pass
        return None

    def get_items_since(self, since: int) -> list[dict]:
        """Get items modified since the given version from user library.

        Args:
            since: The version number to get changes since

        Returns:
            List of changed items
        """
        url = self._get_user_items_url()
        params = {"since": since}
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        return []

    def get_group_items_since(self, group_id: int, since: int) -> list[dict]:
        """Get items modified since the given version from a group's library.

        Args:
            group_id: The group ID
            since: The version number to get changes since

        Returns:
            List of changed items
        """
        url = self._get_group_items_url(group_id)
        params = {"since": since}
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        return []

    def _clean_bibtex_value(self, value: str | None) -> str:
        """Normalize free-text values before rendering them into BibTeX."""

        if not value:
            return ""
        cleaned = re.sub(r"\s+", " ", str(value)).strip()
        return cleaned.replace("\\", "\\\\")

    def _extract_year(self, value: str | None) -> str:
        if not value:
            return ""
        match = re.search(r"\b(19|20)\d{2}\b", str(value))
        return match.group(0) if match else ""

    def _normalize_bibtex_token(self, value: str | None) -> str:
        if not value:
            return ""
        ascii_value = (
            unicodedata.normalize("NFKD", str(value))
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        return re.sub(r"[^A-Za-z0-9]+", "", ascii_value)

    def _creator_to_bibtex_name(self, creator: dict) -> str:
        corporate_name = self._clean_bibtex_value(creator.get("name"))
        if corporate_name:
            return corporate_name

        first_name = self._clean_bibtex_value(creator.get("firstName"))
        last_name = self._clean_bibtex_value(creator.get("lastName"))
        if first_name and last_name:
            return f"{last_name}, {first_name}"
        return last_name or first_name

    def _append_bibtex_field(
        self, fields: list[str], name: str, value: str | None
    ) -> None:
        cleaned = self._clean_bibtex_value(value)
        if cleaned:
            fields.append(f"  {name} = {{{cleaned}}},")

    def _build_bibtex_key(self, item: dict) -> str:
        data = item.get("data", {})
        key = item.get("key", "item")
        creators = data.get("creators", [])
        author_seed = ""
        if creators:
            author_seed = (
                creators[0].get("lastName")
                or creators[0].get("name")
                or creators[0].get("firstName")
                or ""
            )
        title = data.get("title", "")
        year = self._extract_year(data.get("date"))
        title_words = [
            self._normalize_bibtex_token(word)
            for word in re.split(r"\W+", title)
            if len(word) > 2
        ]
        title_seed = "".join(title_words[:2])
        cite_key = f"{self._normalize_bibtex_token(author_seed)}{year}{title_seed}"
        return cite_key or self._normalize_bibtex_token(key) or "item"

    def _bibtex_type_for_item(self, data: dict) -> str:
        item_type = data.get("itemType", "misc")
        if item_type == "thesis":
            thesis_type = (data.get("thesisType") or "").lower()
            if "master" in thesis_type:
                return "mastersthesis"
            return "phdthesis"
        return self._zotero_to_bibtex_type(item_type)

    def item_to_bibtex(self, item: dict) -> str:
        """Convert a Zotero item to a BibTeX entry."""
        data = item.get("data", {})
        key = item.get("key", "")
        bibtex_type = self._bibtex_type_for_item(data)
        cite_key = self._build_bibtex_key(item)
        year = self._extract_year(data.get("date"))

        creators = data.get("creators", [])
        author_types = {
            "author",
            "inventor",
            "programmer",
            "artist",
            "podcaster",
            "presenter",
            "contributor",
        }
        editor_types = {"editor", "seriesEditor"}
        authors = [
            self._creator_to_bibtex_name(c)
            for c in creators
            if (c.get("creatorType") or "author") in author_types
            and self._creator_to_bibtex_name(c)
        ]
        editors = [
            self._creator_to_bibtex_name(c)
            for c in creators
            if c.get("creatorType") in editor_types and self._creator_to_bibtex_name(c)
        ]

        publication = data.get("publicationTitle") or data.get("journalAbbreviation")
        booktitle = (
            data.get("proceedingsTitle")
            or data.get("bookTitle")
            or data.get("conferenceName")
        )
        institution = data.get("institution") or data.get("publisher")
        school = data.get("university") or data.get("publisher")

        fields: list[str] = []
        self._append_bibtex_field(fields, "title", data.get("title"))
        if authors:
            self._append_bibtex_field(fields, "author", " and ".join(authors))
        if editors:
            self._append_bibtex_field(fields, "editor", " and ".join(editors))
        self._append_bibtex_field(fields, "year", year)

        full_date = self._clean_bibtex_value(data.get("date"))
        if full_date and full_date != year:
            self._append_bibtex_field(fields, "date", full_date)

        if bibtex_type == "article":
            self._append_bibtex_field(fields, "journal", publication)
        elif bibtex_type in {"inproceedings", "incollection", "inreference"}:
            self._append_bibtex_field(fields, "booktitle", booktitle or publication)
        elif bibtex_type in {"techreport"}:
            self._append_bibtex_field(fields, "institution", institution)
        elif bibtex_type in {"phdthesis", "mastersthesis"}:
            self._append_bibtex_field(fields, "school", school)
        elif bibtex_type == "misc":
            self._append_bibtex_field(
                fields, "howpublished", data.get("websiteTitle") or publication
            )

        self._append_bibtex_field(fields, "publisher", data.get("publisher"))
        self._append_bibtex_field(fields, "volume", data.get("volume"))
        self._append_bibtex_field(
            fields, "number", data.get("issue") or data.get("seriesNumber")
        )
        self._append_bibtex_field(fields, "pages", data.get("pages"))
        self._append_bibtex_field(fields, "series", data.get("series"))
        self._append_bibtex_field(fields, "edition", data.get("edition"))
        self._append_bibtex_field(fields, "address", data.get("place"))
        self._append_bibtex_field(fields, "doi", data.get("DOI"))
        self._append_bibtex_field(fields, "url", data.get("url"))
        self._append_bibtex_field(fields, "isbn", data.get("ISBN"))
        self._append_bibtex_field(fields, "issn", data.get("ISSN"))

        abstract = self._clean_bibtex_value(data.get("abstractNote"))
        if len(abstract) > 500:
            abstract = abstract[:497] + "..."
        self._append_bibtex_field(fields, "abstract", abstract)
        self._append_bibtex_field(fields, "note", data.get("note"))
        self._append_bibtex_field(fields, "zotero_key", key)

        return f"@{bibtex_type}{{{cite_key},\n" + "\n".join(fields) + "\n}}"

    def _zotero_to_bibtex_type(self, item_type: str) -> str:
        """Map Zotero item types to BibTeX entry types."""
        type_map = {
            "journalArticle": "article",
            "book": "book",
            "bookSection": "incollection",
            "conferencePaper": "inproceedings",
            "thesis": "phdthesis",
            "report": "techreport",
            "webpage": "misc",
            "patent": "misc",
            "statute": "misc",
            "case": "misc",
            "email": "misc",
            "letter": "misc",
            "manuscript": "unpublished",
            "presentation": "misc",
            "radioBroadcast": "misc",
            "tvBroadcast": "misc",
            "videoRecording": "misc",
            "audioRecording": "misc",
            "podcast": "misc",
            "blogPost": "misc",
            "forumPost": "misc",
            "dictionaryEntry": "incollection",
            "encyclopediaArticle": "inreference",
            "newspaperArticle": "article",
            "magazineArticle": "article",
            "attachment": "misc",
            "note": "misc",
        }
        return type_map.get(item_type, "misc")

    def import_bibtex_via_connector(
        self,
        bibtex: str,
        *,
        session_id: str | None = None,
        collection_key: str | None = None,
        timeout_seconds: int | None = None,
    ) -> dict:
        """Import a BibTeX entry via the Zotero Connector local endpoint.

        Zotero Connector listens on the same base URL as the local API (23119 by default)
        but under the `/connector/...` path.

        Args:
            bibtex: BibTeX string to import.
            session_id: Optional connector session id; if omitted a random UUID is used.
            collection_key: Optional collection key to target. If omitted, uses
                Config.ZOTERO_DEFAULT_IMPORT_COLLECTION_KEY when set.
            timeout_seconds: Optional request timeout override.

        Returns:
            Dict containing status, http_status, and response payload.
        """
        import uuid

        if not bibtex or not bibtex.strip():
            return {"status": "error", "message": "BibTeX must be non-empty"}

        if session_id is None:
            session_id = str(uuid.uuid4())

        # Decide collection key: explicit arg wins, then config default.
        if collection_key is None:
            try:
                from .config import config as _cfg

                collection_key = _cfg.ZOTERO_DEFAULT_IMPORT_COLLECTION_KEY or None
            except Exception:
                collection_key = None

        try:
            from .config import config as _cfg

            import_path = getattr(
                _cfg, "ZOTERO_CONNECTOR_IMPORT_PATH", "/connector/import"
            )
            default_timeout = getattr(_cfg, "ZOTERO_CONNECTOR_TIMEOUT_SECONDS", 15)
        except Exception:
            import_path = "/connector/import"
            default_timeout = 15

        timeout = default_timeout if timeout_seconds is None else timeout_seconds

        url = f"{self.api_url}{import_path}"
        params: dict[str, str] = {"session": session_id}
        if collection_key:
            params["collection"] = collection_key

        headers = {
            # This endpoint expects the connector header name (different from Zotero-API-Version)
            "X-Zotero-Connector-API-Version": str(self.API_VERSION),
            "Content-Type": "application/x-bibtex",
        }

        try:
            resp = self.session.post(
                url,
                params=params,
                headers=headers,
                data=bibtex.encode("utf-8"),
                timeout=timeout,
            )
            ok = resp.status_code in (200, 201)
            try:
                payload = resp.json()
            except ValueError:
                payload = resp.text

            if not ok:
                return {
                    "status": "error",
                    "http_status": resp.status_code,
                    "message": "Connector import failed",
                    "response": payload,
                }

            # The connector typically returns a list of imported items with keys.
            imported_keys: list[str] = []
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict) and item.get("key"):
                        imported_keys.append(item["key"])

            return {
                "status": "imported",
                "http_status": resp.status_code,
                "response": payload,
                "imported_keys": imported_keys,
                "session": session_id,
                "collection": collection_key or "",
            }
        except requests.RequestException as e:
            return {
                "status": "error",
                "message": f"Connector import request failed: {e}",
                "session": session_id,
                "collection": collection_key or "",
            }
