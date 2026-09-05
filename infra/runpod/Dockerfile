FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    VIRTUAL_ENV=/opt/oasis/.venv \
    PATH=/opt/oasis/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HF_HOME=/workspace/cache/huggingface \
    TRITON_CACHE_DIR=/workspace/cache/triton

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates python3.12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/oasis

# Install third-party packages before copying frequently changed source files,
# keeping normal code-only rebuilds fast. The Runpod group provides S3 sync.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --group gpu --group runpod --no-install-project

COPY src ./src
COPY data ./data
RUN uv sync --frozen --no-default-groups --group gpu --group runpod --no-editable

CMD ["python", "-m", "oasis.runpod_experiments", "worker"]
