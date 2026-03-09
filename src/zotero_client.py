"""Compatibility shim for older tests.

The project code uses the package-qualified import `semtero.zotero_client`.
Some unit tests patch `zotero_client.ZoteroClient` directly.

This module re-exports `ZoteroClient` so those patches resolve.
"""

from semtero.zotero_client import ZoteroClient  # noqa: F401
