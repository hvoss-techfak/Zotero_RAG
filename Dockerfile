# ZoteroRAG - MCP Server for Zotero
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/

# Install Python dependencies via uv
RUN pip install uv && uv sync

# Create data directories
RUN mkdir -p /app/data/pdfs /app/data/vector_store

# Expose MCP server port (default 23120)
EXPOSE 23120

# Run the MCP server
CMD ["uv", "run", "python", "-m", "src.zoterorag.mcp_server"]