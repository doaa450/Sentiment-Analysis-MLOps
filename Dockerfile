FROM python:3.14-slim AS builder
RUN pip install --no-cache-dir uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
WORKDIR /build
# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
# Then install the package itself (non-editable, so the venv is self-contained)
COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.14-slim AS runtime
RUN useradd --create-home --uid 1000 appuser \
    && install -d -o appuser -g appuser /home/appuser/.cache/huggingface
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    SENTIMENT_MODEL_NAME=DoaaMoh/arabert-sentiment \
    SENTIMENT_MODEL_VERSION=v0.2.0
USER appuser
WORKDIR /app
EXPOSE 8000
# Long start period: the model is downloaded and loaded at startup
HEALTHCHECK --interval=30s --timeout=3s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "arabic_sentiment.api:app", "--host", "0.0.0.0", "--port", "8000"]