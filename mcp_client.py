#!/usr/bin/env python3
"""MCP Client - Test script to call ZoteroRAG MCP server methods.

Usage:
    uv run mcp_client.py --search QUERY     # Connect via SSE to running server
    uv run mcp_client.py --library          # Show library items  
    uv run mcp_client.py --status           # Show embedding status
    uv run mcp_client.py --stdio            # Use stdio (spawns new server)

This script can connect to an already-running MCP server via SSE,
or spawn a new server process via stdio (legacy behavior).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastmcp import Client  # type: ignore
from fastmcp.client.transports.sse import SSETransport  # type: ignore
from fastmcp.client.transports.stdio import PythonStdioTransport  # type: ignore


def extract_tool_result(result) -> list | dict | None:
    """Extract actual data from FastMCP CallToolResult.
    
    FastMCP's call_tool() returns a CallToolResult object with content
    wrapped in TextContent. This helper extracts the actual data.
    
    Args:
        result: The CallToolResult from call_tool()
        
    Returns:
        Parsed list/dict from the tool, or None if empty/error
    """
    # Check for error results
    if hasattr(result, 'is_error') and result.is_error:
        return None
    
    # Extract content - can be in multiple forms
    if hasattr(result, 'content') and result.content:
        text_content = result.content[0]
        
        # If it's a TextContent with JSON string, parse it
        if hasattr(text_content, 'text'):
            try:
                parsed = json.loads(text_content.text)
                return parsed
            except (json.JSONDecodeError, TypeError):
                # Not JSON, return as-is or handle differently
                return text_content.text
        
        # If it's already a dict/list (structured content)
        if hasattr(text_content, 'data'):
            return text_content.data
    
    return None


def format_json(data: dict | list) -> str:
    """Format data as pretty JSON."""
    return json.dumps(data, indent=2, ensure_ascii=False)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print('=' * 60)


async def run_library_items(client: Client, limit: int = 10):
    """Get and display library items."""
    print_section("Library Items")

    try:
        result = await client.call_tool("get_library_items", {"limit": limit})
        items = extract_tool_result(result)

        if not items:
            print("No items found in library.")
            return

        # Ensure items is a list
        if not isinstance(items, list):
            print(f"Unexpected result type: {type(items)}")
            return

        for i, item in enumerate(items, 1):
            print(f"\n[{i}] {item.get('title', 'Untitled')}")
            authors = item.get('authors', [])
            if authors:
                print(f"    Authors: {', '.join(authors)}")
            date = item.get('date')
            if date:
                print(f"    Date: {date}")
            key = item.get('key', '')
            if key:
                print(f"    Key: {key}")

        print(f"\nTotal: {len(items)} items")

    except Exception as e:
        print(f"Error fetching library items: {e}")


async def run_search(client: Client, query: str):
    """Search documents and display results."""
    print_section(f"Search Results for: '{query}'")

    try:
        result = await client.call_tool(
            "search_documents",
            {
                "query": query,
                "top_sentences": 10,
                "min_relevance": 0.0,
                "require_cited_bibtex":True,
            },
        )

        results = extract_tool_result(result)

        if not results:
            print("No results found.")
            return

        # Ensure results is a list
        if not isinstance(results, list):
            print(f"Unexpected result type: {type(results)}")
            print(f"Raw result: {result}")
            return

        print(f"Found {len(results)} result(s):\n")

        for i, r in enumerate(results, 1):
            print(r)
            doc_title = r.get('document_title', 'Untitled Document')
            section_title = r.get('section_title', '')
            content = r.get('text', '')[:300] if r.get('text') else ''

            # Show all available metadata
            zotero_key = r.get('zotero_key', '')
            authors = r.get('authors', [])
            date = r.get('date', '')
            bibtex = r.get('bibtex', '')
            item_type = r.get('item_type', '')

            print(f"[{i}] Document: {doc_title}")
            if zotero_key:
                print(f"    Zotero Key: {zotero_key}")
            if authors:
                print(f"    Authors: {', '.join(authors) if isinstance(authors, list) else authors}")
            if date:
                print(f"    Date: {date}")
            if item_type:
                print(f"    Type: {item_type}")
            if section_title:
                print(f"    Section: {section_title}")

            # Show relevance scores
            rel_score = r.get('relevance_score', 0)
            rerank_score = r.get('rerank_score', 0)
            print(f"    Relevance: {rel_score:.4f} | Rerank: {rerank_score:.4f}")

            if content:
                print(f"    Content: {content}...")

            # Show metadata if available
            if bibtex:
                print(f"    BibTeX: Available ({len(bibtex)} chars)")

            print()

        return results

    except Exception as e:
        print(f"Error searching documents: {e}")
        import traceback
        traceback.print_exc()
        return []


async def run_embedding_status(client: Client):
    """Get and display embedding status."""
    print_section("Embedding Status")

    try:
        result = await client.call_tool("get_embedding_status", {})
        status = extract_tool_result(result)

        if not status or not isinstance(status, dict):
            print(f"Unexpected result type: {type(status)}")
            return

        # Format as nice output
        total_sections = status.get('total_sections', 0)
        total_sentences = status.get('total_sentence_windows', 0)
        documents = status.get('documents', [])

        print(f"Total sections embedded: {total_sections}")
        print(f"Total sentence windows indexed: {total_sentences}")
        print(f"Documents with embeddings: {len(documents)}")

        if documents:
            print("\nEmbedded document keys:")
            for key in documents[:20]:
                print(f"  - {key}")
            if len(documents) > 20:
                print(f"  ... and {len(documents) - 20} more")

    except Exception as e:
        print(f"Error getting embedding status: {e}")


async def run_documents_with_pdfs(client: Client):
    """Get documents with PDFs."""
    print_section("Documents with PDFs")

    try:
        result = await client.call_tool("get_documents_with_pdfs", {})
        docs = extract_tool_result(result)

        if not docs:
            print("No documents with PDFs found.")
            return

        # Ensure docs is a list
        if not isinstance(docs, list):
            print(f"Unexpected result type: {type(docs)}")
            return

        for doc in docs:
            key = doc.get('key', '')
            title = doc.get('title', 'Untitled')
            has_pdf = doc.get('has_pdf', False)

            status = "\u2713 Has PDF" if has_pdf else "\u2717 No PDF"
            print(f"  {key}: {title[:50]}... [{status}]")

        print(f"\nTotal: {len(docs)} documents with PDFs")

    except Exception as e:
        print(f"Error getting documents with PDFs: {e}")


async def main():
    parser = argparse.ArgumentParser(description="ZoteroRAG MCP Client")
    parser.add_argument(
        "--search", "-s",
        type=str,
        help="Search query"
    )
    parser.add_argument(
        "--library", "-l",
        action="store_true",
        help="Show library items"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show embedding status"
    )
    parser.add_argument(
        "--pdfs",
        action="store_true",
        help="Show documents with PDFs"
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Run all checks (library, status, pdfs)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit for library items (default: 10)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=23120,
        help="Port for SSE/HTTP connection (default: 23120)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for SSE/HTTP connection (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Use stdio transport instead of SSE (spawns new server process)"
    )

    args = parser.parse_args()

    # If no arguments, show help
    if not any([args.search, args.library, args.status, args.pdfs, args.all]):
        parser.print_help()
        return

    # Create the appropriate transport based on mode
    if args.stdio:
        # Legacy behavior: spawn a new server process via stdio
        server_script = Path(__file__).parent / "main.py"
        transport = PythonStdioTransport(script_path=server_script, python_cmd=sys.executable)
        print(f"Using stdio transport (spawning server)...")
    else:
        # New behavior: connect to already-running server via SSE
        url = f"http://{args.host}:{args.port}/sse"
        transport = SSETransport(url=url)
        print(f"Connecting to {url}...")

    async with Client(transport) as client:
        try:
            if args.all or args.library:
                await run_library_items(client, limit=args.limit)

            if args.all or args.status:
                await run_embedding_status(client)

            if args.all or args.pdfs:
                await run_documents_with_pdfs(client)

            if args.search:
                results = await run_search(client, args.search)

                if args.json and results:
                    print("\n--- JSON Output ---")
                    print(format_json(results))

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

