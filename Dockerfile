FROM python:3.11-slim

# MULTILINGUAL=1 adds CPU-only torch and sentence-transformers for the
# multilingual retrievers; the default 0 keeps the image small (lexical and
# hashing retrievers only).
ARG MULTILINGUAL=0

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt \
    && if [ "$MULTILINGUAL" = "1" ]; then \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
        && pip install --no-cache-dir "sentence-transformers>=3.0"; \
    fi

COPY src ./src
COPY data ./data

RUN useradd --create-home rag && mkdir -p /home/rag/.cache && chown rag /home/rag/.cache
USER rag
ENV HF_HOME=/home/rag/.cache/huggingface

EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
