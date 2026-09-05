# syntax=docker/dockerfile:1.7

# Runpod pre-caches its official PyTorch layers on GPU hosts.  This base also
# supplies Python 3.12, CUDA 12.8, and torch 2.8, so the application layer must
# not install a second copy of torch or its CUDA wheel stack.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/oasis/.venv \
    PATH=/opt/oasis/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HF_HOME=/workspace/cache/huggingface \
    TRITON_CACHE_DIR=/workspace/cache/triton

WORKDIR /opt/oasis

# The venv can see the base image's pinned torch/CUDA installation.  Export the
# fully locked application dependency graph, pruning torch and torchvision so
# uv cannot duplicate the multi-gigabyte CUDA runtime in this custom layer.
RUN python -m venv --system-site-packages "${VIRTUAL_ENV}"

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_CACHE_DIR=/root/.cache/uv uv export \
      --frozen \
      --no-default-groups \
      --group gpu \
      --group runpod \
      --no-emit-project \
      --prune torch \
      --prune torchvision \
      --output-file /tmp/oasis-requirements.txt \
    && UV_CACHE_DIR=/root/.cache/uv uv pip install \
      --python "${VIRTUAL_ENV}/bin/python" \
      --no-deps \
      --requirement /tmp/oasis-requirements.txt \
    && rm /tmp/oasis-requirements.txt

COPY src ./src
COPY data ./data
COPY infra/runpod/osrm-cache ./osrm-cache
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_CACHE_DIR=/root/.cache/uv uv pip install \
      --python "${VIRTUAL_ENV}/bin/python" \
      --no-deps \
      .

# Catch base-image drift and missing runtime dependencies during the GitHub
# build, before a paid Pod is allocated. GPU availability is checked by the
# worker after Runpod starts the container.
RUN python -c "import boto3, geopandas, ortools, rasterio, torch, torchvision, transformers; from transformers import AutoModelForMultimodalLM, Gemma4Config, Gemma4Processor; assert torch.__version__.startswith('2.8.'), torch.__version__; assert torchvision.__version__.startswith('0.23.'), torchvision.__version__; assert AutoModelForMultimodalLM._model_mapping[Gemma4Config].__name__ == 'Gemma4ForConditionalGeneration'; print({'torch': torch.__version__, 'torchvision': torchvision.__version__, 'cuda': torch.version.cuda, 'transformers': transformers.__version__, 'gemma4_processor': Gemma4Processor.__name__})"

# Exercise every dataset through the actual registry without a GPU, model
# download, geocoding network call, or credentials. Fail the build on bad wiring.
RUN python -m oasis.registry_smoke --rows 1 \
      --osrm-cache /opt/oasis/osrm-cache --output /tmp/oasis-registry-build-check

CMD ["python", "-m", "oasis.runpod_experiments", "worker"]
