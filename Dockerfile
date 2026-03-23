FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app

RUN python -m venv /opt/venv

ARG PRELOAD_TROCR_MODEL=false

COPY backend/requirements-linux.txt /tmp/requirements-linux.txt
COPY backend/requirements-base.txt /tmp/requirements-base.txt
COPY backend/scripts/warm_trocr_cache.py /tmp/warm_trocr_cache.py
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements-linux.txt

RUN mkdir -p /opt/huggingface \
    && if [ "$PRELOAD_TROCR_MODEL" = "true" ]; then HF_HOME=/opt/huggingface python /tmp/warm_trocr_cache.py; fi

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONPATH=/app
ENV XDG_CACHE_HOME=/home/app/.cache
ENV HF_HOME=/home/app/.cache/huggingface
ENV CHROMA_PERSIST_DIRECTORY=/home/app/data/chroma

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && addgroup --system app \
    && adduser --system --ingroup app --home /home/app app \
    && mkdir -p /home/app/.cache/huggingface /home/app/data/chroma /app/db \
    && chown -R app:app /home/app /app/db

COPY --from=builder --chown=app:app /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/huggingface /home/app/.cache/huggingface
COPY --chown=app:app backend /app/backend
COPY --chown=app:app README.md /app/README.md

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "import sys, urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health'); sys.exit(0)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
