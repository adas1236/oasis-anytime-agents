"""Authenticated OASIS model-worker service contracts."""

from typing import Any

from oasis.model_worker.schemas import MODEL_WORKER_SCHEMA_VERSION


def create_model_worker_app(*args: Any, **kwargs: Any) -> Any:
    """Import the FastAPI worker lazily to keep schema/runtime imports acyclic."""

    from oasis.model_worker.app import create_model_worker_app as create

    return create(*args, **kwargs)


__all__ = ["MODEL_WORKER_SCHEMA_VERSION", "create_model_worker_app"]
