# FaceGate recognition API - CPU image.
#
#   docker build -t facegate:cpu .
#   docker run --rm -p 127.0.0.1:8000:8000 -v facegate-data:/data facegate:cpu
#
# Model weights (~190 MB) are downloaded into the /data volume on first start
# and verified against pinned SHA-256 digests. They are not baked into the
# image because the InsightFace weights are licensed for non-commercial use.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FACEGATE_HOME=/data

WORKDIR /app

COPY requirements-server.txt .
RUN pip install -r requirements-server.txt

COPY facegate ./facegate

RUN useradd --create-home --uid 10001 facegate \
    && mkdir /data && chown facegate:facegate /data
USER facegate
VOLUME /data
EXPOSE 8000

# start-period covers the first-run model download
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8).status == 200 else 1)"

CMD ["python", "-m", "facegate", "serve", "--host", "0.0.0.0", "--port", "8000"]
