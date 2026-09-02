FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HF_HOME=/studio/storage/model-cache \
    JDG_DATABASE_URL=sqlite:////studio/runtime/jdg_studio.db \
    JDG_ADAPTER_PATH=/studio/storage/adapters/dev \
    JDG_MODEL_AUTOLOAD=true \
    JDG_ALLOW_MODEL_DOWNLOAD=true
WORKDIR /studio

RUN groupadd --system studio && useradd --system --gid studio --create-home studio
COPY pyproject.toml README.md LICENSE ./
COPY app ./app
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5,<3" \
    && python -m pip install .
COPY configs ./configs
COPY data/samples ./data/samples
COPY storage/adapters/dev ./storage/adapters/dev
COPY storage/evaluation_results.json ./storage/evaluation_results.json
RUN mkdir -p /studio/runtime /studio/storage/exports /studio/storage/mlruns /studio/storage/model-cache \
    && chown -R studio:studio /studio

USER studio
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/api/v1/health', timeout=4)"
CMD ["sh", "-c", "python -m app.cli serve --host 0.0.0.0 --port ${PORT:-8000}"]
