# SemTero - MCP Server for Zotero
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY main.py mcp_client.py ./
COPY src/ ./src/
COPY webui/ ./webui/
COPY .env.example ./

# Install Python dependencies via uv
RUN uv sync --frozen --no-dev
# Create data directories
RUN mkdir -p /app/data/pdfs /app/data/vector_store

# Expose MCP server port (default 23120)
EXPOSE 23120 23121

# Run the MCP server
CMD ["uv", "run", "--no-sync", "python", "main.py", "--transport", "streamable-http", "--host", "0.0.0.0", "--webui-host", "0.0.0.0", "--port", "23120", "--webui-port", "23121"]
