FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends redis-server \
  && rm -rf /var/lib/apt/lists/*

COPY runtime/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV FASTEMBED_CACHE_PATH=/opt/fastembed_cache
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

# Block all outbound model downloads at runtime — airgap safe.
# Model is already baked in above; these prevent any stray SDK calls.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

WORKDIR /runtime
COPY runtime /runtime

EXPOSE 6379 8080

RUN mkdir -p /runtime-data /data/redis

ENTRYPOINT ["python", "/runtime/entrypoint.py"]
