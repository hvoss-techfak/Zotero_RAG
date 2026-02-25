#!/usr/bin/env python3
"""Diagnostic script to check PDF availability from Zotero.

This helps identify why only ~90 PDFs are being embedded instead of 3000+.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pathlib import Path
from zoterorag.config import Config
from zoterorag.zotero_client import ZoteroClient
from zoterorag.vector_store import VectorStore

def main():
    Config.ensure_dirs()
    
    print(f"Connecting to Zotero at {Config.ZOTERO_API_URL}")
    client = ZoteroClient(api_url=Config.ZOTERO_API_URL)
    
    print("=" * 60)
    print("ZoteroRAG Diagnostic Report")
    print("=" * 60)
    
    # Check connection
    print("\n1. Checking Zotero API connection...")
    if not client.check_connection():
        print("   ERROR: Cannot connect to Zotero API at", Config.ZOTERO_API_URL)
        print("   Make sure Zotero is running with the local connector enabled.")
        return 1
    
    print("   ✓ Connected to Zotero")
    
    # Get user items count
    print("\n2. Checking library sizes...")
    
    try:
        user_count = client.get_total_items_count()
        print(f"   User library: {user_count} items")
    except Exception as e:
        print(f"   Error getting user count: {e}")
        user_count = -1
    
    # Get groups
    groups = client.get_groups()
    print(f"\n3. Found {len(groups)} group(s):")
    
    total_group_items = 0
    for g in groups[:10]:  # Show first 10
        gid = g.get("id", "unknown")
        name = g.get("data", {}).get("name", "Unnamed")
        
        try:
            # Get count via first page header
            items = client.get_group_items(gid, limit=1)
            from requests import Response
            # This won't work directly since get_group_items doesn't expose headers easily
            print(f"   - Group {gid}: {name}")
        except Exception as e:
            print(f"   - Group {gid}: {name} (error: {e})")
    
    if len(groups) > 10:
        print(f"   ... and {len(groups) - 10} more groups")
    
    # Count PDFs in local cache
    pdf_dir = Config.PDF_CACHE_PATH
    local_pdfs = list(pdf_dir.glob("*.pdf"))
    print(f"\n4. Local PDF cache:")
    print(f"   Found {len(local_pdfs)} PDFs in {pdf_dir}")
    
    # Get all documents with PDFs from Zotero (this is the key check)
    print("\n5. Fetching ALL items with PDFs from Zotero...")
    print("   (This may take a while for large libraries...)")
    
    docs_with_pdfs = []
    user_docs = 0
    group_docs = 0
    
    try:
        for i, doc in enumerate(client.get_documents_with_pdfs()):
            if i % 100 == 0 and i > 0:
                print(f"   ... processed {i} items with PDFs ...")
            
            docs_with_pdfs.append(doc)
            if doc.group_id is None:
                user_docs += 1
            else:
                group_docs += 1
        
        print(f"\n   Total documents with PDFs: {len(docs_with_pdfs)}")
        print(f"   - User library: {user_docs}")
        print(f"   - Group libraries: {group_docs}")
        
    except Exception as e:
        print(f"   ERROR fetching documents: {e}")
        return 1
    
    # Compare with local
    local_keys = set(pdf.stem for pdf in local_pdfs)
    zotero_keys = set(doc.zotero_key for doc in docs_with_pdfs)
    
    missing = zotero_keys - local_keys
    extra_local = local_keys - zotero_keys
    
    print(f"\n6. Comparison:")
    print(f"   In Zotero: {len(zotero_keys)}")
    print(f"   Locally cached: {len(local_keys)}")
    print(f"   Missing from cache: {len(missing)}")
    print(f"   Extra local files (not in Zotero): {len(extra_local)}")
    
    if missing:
        print(f"\n7. First 20 missing PDFs:")
        for key in sorted(missing)[:20]:
            # Find the doc to show title
            doc = next((d for d in docs_with_pdfs if d.zotero_key == key), None)
            title = doc.title[:40] + "..." if doc and len(doc.title) > 40 else (doc.title if doc else "Unknown")
            print(f"   - {key}: {title}")
        
        if len(missing) > 20:
            print(f"   ... and {len(missing) - 20} more")
    
    # Already imported at top
    vs = VectorStore(str(Config.VECTOR_STORE_DIR))
    embedded = vs.get_embedded_documents()
    
    print(f"\n8. Embedding status:")
    print(f"   Documents in vector store: {len(embedded)}")
    
    zero_sections = [k for k, v in embedded.items() if v == 0]
    nonzero_sections = [k for k, v in embedded.items() if v > 0]
    
    print(f"   - Successfully embedded (>0 sections): {len(nonzero_sections)}")
    print(f"   - Failed (0 sections): {len(zero_sections)}")
    
    if zero_sections:
        print("\n   Documents that failed to embed:")
        for key in sorted(zero_sections)[:10]:
            doc = next((d for d in docs_with_pdfs if d.zotero_key == key), None)
            title = doc.title[:40] + "..." if doc and len(doc.title) > 40 else (doc.title if doc else "Unknown")
            print(f"   - {key}: {title}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Zotero has {len(docs_with_pdfs)} documents with PDFs")
    print(f"  Local cache has {len(local_pdfs)} PDFs")  
    print(f"  Missing: {len(missing)} PDFs need to be downloaded")
    print(f"  Already embedded: {len(nonzero_sections)} documents")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())