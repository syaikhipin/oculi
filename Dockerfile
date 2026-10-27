FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/app/data
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[faiss]" \
    && useradd --create-home --uid 10001 oculi \
    && mkdir -p /app/data /app/models \
    && chown -R oculi:oculi /app

USER oculi
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/livez')"
CMD ["oculi", "serve", "--host", "0.0.0.0", "--port", "8080"]
