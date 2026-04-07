"""Core check function - the main entry point."""

import logging
from typing import Optional

from mmcheck.models import ModelInfo
from mmcheck.registry import lookup
from mmcheck.hub import fetch_from_hub
from mmcheck.vllm_models import enrich_with_vllm

logger = logging.getLogger(__name__)


def check(model: str, offline: bool = False, token: str | None = None, timeout: int = 10) -> ModelInfo:
    """Check if a model supports multimodal inputs.

    Resolution order:
    1. Built-in registry (instant, no network)
    2. HuggingFace Hub config.json (requires network)
    3. Fallback to text-only unknown

    Args:
        model: Model name or HuggingFace model ID (e.g., "gpt-4o", "google/gemma-3-27b-it")
        offline: If True, skip HuggingFace Hub lookup
        token: Optional HuggingFace token for gated models (or set HF_TOKEN env var)
        timeout: Timeout in seconds for HuggingFace API calls

    Returns:
        ModelInfo with detected capabilities
    """
    # 1. Try built-in registry first
    info = lookup(model)
    if info is not None:
        return info

    # 2. Try HuggingFace Hub
    if not offline:
        info = fetch_from_hub(model, token=token, timeout=timeout)
        if info is not None:
            return enrich_with_vllm(info)

    # 3. Fallback
    return ModelInfo(name=model, source="unknown")
