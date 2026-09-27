# uae-fwa-engine
#
# A single image that runs the application, the validation run and the test
# suite. Built deliberately WITHOUT network access at runtime: the artefact's
# central claim is that it needs no API key and no external service, and an
# image that quietly reaches out would undermine that claim rather than
# demonstrate it.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # The offline provider is the default in code; setting it here makes the
    # container's behaviour explicit to anyone reading the image.
    FWA_LLM_PROVIDER=offline_deterministic

WORKDIR /app

# Build tooling for the scientific stack, removed again in the same layer so it
# does not ship in the image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# Dependencies first, so a source change does not reinstall the stack.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[dev]" \
 && apt-get purge -y --auto-remove build-essential

COPY config/ ./config/
COPY rules/ ./rules/
COPY data/ ./data/
COPY app/ ./app/
COPY tests/ ./tests/
COPY tools/ ./tools/
COPY docs/ ./docs/

# Run as a non-root user. The reports directory is the one writable path the
# validation run needs.
RUN useradd --create-home --uid 10001 fwa \
 && mkdir -p /app/reports /app/data/synthetic_documents \
 && chown -R fwa:fwa /app
USER fwa

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/Home.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
