FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Bake the two local models into the image
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('BAAI/bge-reranker-v2-m3')"

COPY baseline ./baseline
COPY src ./src
COPY data ./data

# Build the search index from the PDFs
RUN python -m src.pipeline_v1 build

ENV HF_HUB_OFFLINE=1 \
    LLM_PROVIDER=anthropic \
    LLM_MODEL=claude-haiku-4-5-20251001

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]