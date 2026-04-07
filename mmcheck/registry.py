"""Built-in registry for popular API and open-source models."""

from mmcheck.models import ModelInfo

# Static registry of well-known models and their capabilities.
# Format: model_id -> ModelInfo kwargs
_REGISTRY: dict = {
    # OpenAI
    "gpt-4o": dict(
        multimodal=True,
        input_modalities=["text", "image", "audio"],
        output_modalities=["text", "audio"],
        source="registry",
    ),
    "gpt-4o-mini": dict(
        multimodal=True,
        input_modalities=["text", "image"],
        output_modalities=["text"],
        source="registry",
    ),
    "gpt-4-vision-preview": dict(
        multimodal=True,
        input_modalities=["text", "image"],
        output_modalities=["text"],
        source="registry",
    ),
    "gpt-4.1": dict(
        multimodal=True,
        input_modalities=["text", "image"],
        output_modalities=["text"],
        source="registry",
    ),
    "o4-mini": dict(
        multimodal=True,
        input_modalities=["text", "image"],
        output_modalities=["text"],
        source="registry",
    ),
    # Anthropic
    "claude-sonnet-4-20250514": dict(
        multimodal=True,
        input_modalities=["text", "image"],
        output_modalities=["text"],
        source="registry",
    ),
    "claude-3-5-sonnet": dict(
        multimodal=True,
        input_modalities=["text", "image"],
        output_modalities=["text"],
        source="registry",
    ),
    "claude-3-opus": dict(
        multimodal=True,
        input_modalities=["text", "image"],
        output_modalities=["text"],
        source="registry",
    ),
    "claude-3-haiku": dict(
        multimodal=True,
        input_modalities=["text", "image"],
        output_modalities=["text"],
        source="registry",
    ),
    # Google
    "gemini-2.5-pro": dict(
        multimodal=True,
        input_modalities=["text", "image", "audio", "video"],
        output_modalities=["text"],
        source="registry",
    ),
    "gemini-2.5-flash": dict(
        multimodal=True,
        input_modalities=["text", "image", "audio", "video"],
        output_modalities=["text"],
        source="registry",
    ),
    "gemini-2.0-flash": dict(
        multimodal=True,
        input_modalities=["text", "image", "audio", "video"],
        output_modalities=["text", "image", "audio"],
        source="registry",
    ),
    "gemini-1.5-pro": dict(
        multimodal=True,
        input_modalities=["text", "image", "audio", "video"],
        output_modalities=["text"],
        source="registry",
    ),
    "gemini-1.5-flash": dict(
        multimodal=True,
        input_modalities=["text", "image", "audio", "video"],
        output_modalities=["text"],
        source="registry",
    ),
    # Meta Llama
    "llama-3.2-11b-vision": dict(
        multimodal=True,
        input_modalities=["text", "image"],
        output_modalities=["text"],
        source="registry",
    ),
    "llama-3.2-90b-vision": dict(
        multimodal=True,
        input_modalities=["text", "image"],
        output_modalities=["text"],
        source="registry",
    ),
    "llama-4-scout": dict(
        multimodal=True,
        input_modalities=["text", "image"],
        output_modalities=["text"],
        source="registry",
    ),
    "llama-4-maverick": dict(
        multimodal=True,
        input_modalities=["text", "image"],
        output_modalities=["text"],
        source="registry",
    ),
    # Mistral
    "pixtral-large": dict(
        multimodal=True,
        input_modalities=["text", "image"],
        output_modalities=["text"],
        source="registry",
    ),
    "pixtral-12b": dict(
        multimodal=True,
        input_modalities=["text", "image"],
        output_modalities=["text"],
        source="registry",
    ),
    # Qwen
    "qwen-vl-max": dict(
        multimodal=True,
        input_modalities=["text", "image", "video"],
        output_modalities=["text"],
        source="registry",
    ),
    "qwen2.5-omni": dict(
        multimodal=True,
        input_modalities=["text", "image", "audio", "video"],
        output_modalities=["text", "audio"],
        source="registry",
    ),
    # Google open models
    "gemma-3-27b-it": dict(
        multimodal=True,
        input_modalities=["text", "image"],
        output_modalities=["text"],
        source="registry",
    ),
    "gemma-3-12b-it": dict(
        multimodal=True,
        input_modalities=["text", "image"],
        output_modalities=["text"],
        source="registry",
    ),
    "gemma-3-4b-it": dict(
        multimodal=True,
        input_modalities=["text", "image"],
        output_modalities=["text"],
        source="registry",
    ),
    "gemma-4-27b-it": dict(
        multimodal=True,
        input_modalities=["text", "image", "video"],
        output_modalities=["text"],
        source="registry",
    ),
    "gemma-4-12b-it": dict(
        multimodal=True,
        input_modalities=["text", "image", "video"],
        output_modalities=["text"],
        source="registry",
    ),
    "gemma-4-4b-it": dict(
        multimodal=True,
        input_modalities=["text", "image", "audio", "video"],
        output_modalities=["text"],
        source="registry",
    ),
    "gemma-4-2b-it": dict(
        multimodal=True,
        input_modalities=["text", "image", "audio", "video"],
        output_modalities=["text"],
        source="registry",
    ),
    # Text-only baselines
    "gpt-4": dict(source="registry"),
    "gpt-3.5-turbo": dict(source="registry"),
    "claude-3-5-sonnet-text": dict(source="registry"),
    "llama-3.1-70b": dict(source="registry"),
    "llama-3.1-8b": dict(source="registry"),
    "mistral-large": dict(source="registry"),
    "gemma-2-27b-it": dict(source="registry"),
}


def lookup(name: str) -> ModelInfo | None:
    """Look up a model in the built-in registry.

    Tries exact match first, then partial/suffix matching.
    """
    lower = name.lower()

    # Exact match
    if lower in _REGISTRY:
        return ModelInfo(name=name, **_REGISTRY[lower])

    # Try matching without provider prefix (e.g., "openai/gpt-4o" -> "gpt-4o")
    short = lower.split("/")[-1] if "/" in lower else None
    if short and short in _REGISTRY:
        return ModelInfo(name=name, **_REGISTRY[short])

    # Fuzzy suffix match (e.g., "meta-llama/Llama-3.2-11B-Vision-Instruct" -> "llama-3.2-11b-vision")
    for key, info in _REGISTRY.items():
        if key in lower.replace("-instruct", "").replace("-chat", ""):
            return ModelInfo(name=name, **info)

    return None
