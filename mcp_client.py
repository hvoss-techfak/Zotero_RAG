#!/usr/bin/env python3
"""MCP Client - Test script to call ZoteroRAG MCP server methods.

Usage:
    uv run scripts/mcp_client.py [--search QUERY] [--library] [--status]

This script directly calls the MCPZoteroServer methods and displays results nicely.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zoterorag.config import Config
from zoterorag.mcp_server import MCPZoteroServer


def format_json(data: dict | list) -> str:
    """Format data as pretty JSON."""
    return json.dumps(data, indent=2, ensure_ascii=False)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print('=' * 60)


async def run_library_items(server: MCPZoteroServer, limit: int = 10):
    """Get and display library items."""
    print_section("Library Items")
    
    try:
        items = await server.get_library_items(limit=limit)
        
        if not items:
            print("No items found in library.")
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


async def run_search(server: MCPZoteroServer, query: str):
    """Search documents and display results."""
    print_section(f"Search Results for: '{query}'")
    
    try:
        results = await server.search_documents(
            query=query,
            top_sections=5,
            top_sentences_per_section=3
        )
        
        if not results:
            print("No results found.")
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


async def run_embedding_status(server: MCPZoteroServer):
    """Get and display embedding status."""
    print_section("Embedding Status")
    
    try:
        status = await server.get_embedding_status()
        
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


async def run_documents_with_pdfs(server: MCPZoteroServer):
    """Get documents with PDFs."""
    print_section("Documents with PDFs")
    
    try:
        docs = await server.get_documents_with_pdfs()
        
        if not docs:
            print("No documents with PDFs found.")
            return
            
        for doc in docs:
            key = doc.get('key', '')
            title = doc.get('title', 'Untitled')
            has_pdf = doc.get('has_pdf', False)
            
            status = "✓ Has PDF" if has_pdf else "✗ No PDF"
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
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any([args.search, args.library, args.status, args.pdfs, args.all]):
        parser.print_help()
        return
    
    # Initialize server
    print("Initializing MCP server...")
    config = Config()
    server = MCPZoteroServer(config)
    
    # Run requested operations
    try:
        if args.all or args.library:
            await run_library_items(server, limit=args.limit)
            
        if args.all or args.status:
            await run_embedding_status(server)
            
        if args.all or args.pdfs:
            await run_documents_with_pdfs(server)
            
        if args.search:
            results = await run_search(server, args.search)
            
            if args.json and results:
                print("\n--- JSON Output ---")
                print(format_json(results))
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())