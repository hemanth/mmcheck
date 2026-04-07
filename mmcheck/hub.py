"""HuggingFace Hub config inspector for detecting multimodal capabilities."""

import logging
import os
from typing import Optional

import requests

from mmcheck.models import ModelInfo

logger = logging.getLogger(__name__)

HF_API = "https://huggingface.co/api/models"
HF_CONFIG_URL = "https://huggingface.co/{model_id}/resolve/main/config.json"

# Keys in config.json that indicate multimodal capabilities
_VISION_KEYS = [
    "vision_config",
    "vision_tower",
    "visual",
    "image_text_hidden_size",
    "vision_encoder",
    "visual_encoder",
    "image_size",
    "vit_model",
    "mm_vision_tower",
    "image_processor",
    "pixel_shuffle_ratio",
]

_AUDIO_KEYS = [
    "audio_config",
    "audio_encoder",
    "audio_model",
    "whisper_model",
    "speech_encoder",
    "audio_tower",
]

_VIDEO_KEYS = [
    "video_config",
    "video_encoder",
    "temporal_encoder",
    "video_tower",
]

# Architecture names that are known multimodal
_MM_ARCHITECTURES = {
    # Vision-Language
    "llava": ["image"],
    "llava_next": ["image", "video"],
    "llava_onevision": ["image", "video"],
    "cogvlm": ["image"],
    "cogvlm2": ["image"],
    "internvl": ["image"],
    "internvl2": ["image"],
    "qwen2_vl": ["image", "video"],
    "qwen2_5_vl": ["image", "video"],
    "qwen2_audio": ["audio"],
    "qwen_omni": ["image", "audio", "video"],
    "mllama": ["image"],
    "gemma3": ["image"],
    "gemma4": ["image", "video"],
    "pixtral": ["image"],
    "phi3_v": ["image"],
    "phi4_mm": ["image"],
    "idefics": ["image"],
    "idefics2": ["image"],
    "idefics3": ["image"],
    "fuyu": ["image"],
    "paligemma": ["image"],
    "chameleon": ["image"],
    "aria": ["image"],
    "molmo": ["image"],
    "got_ocr2": ["image"],
    "minicpm_v": ["image", "video"],
    "minicpm_o": ["image", "audio", "video"],
    "janus": ["image"],
    "deepseek_vl2": ["image"],
    "emu3": ["image"],
    "whisper": ["audio"],
    "ultravox": ["audio"],
    # Encoder-decoder multimodal
    "florence": ["image"],
    "blip": ["image"],
    "blip2": ["image"],
    "git": ["image"],
    "salesforce": ["image"],
}


def _detect_modalities_from_config(config: dict) -> tuple[list[str], Optional[str], Optional[str]]:
    """Detect input modalities from a HuggingFace config.json.

    Returns (extra_modalities, architecture, model_type).
    """
    modalities = set()
    arch = None
    model_type = config.get("model_type", None)

    # Check architectures field
    architectures = config.get("architectures", [])
    if architectures:
        arch = architectures[0]

    # Check model_type against known multimodal types
    if model_type:
        mt_lower = model_type.lower().replace("-", "_")
        for key, mods in _MM_ARCHITECTURES.items():
            if key in mt_lower:
                modalities.update(mods)
                break

    # Check architecture class names
    for a in architectures:
        a_lower = a.lower()
        for key, mods in _MM_ARCHITECTURES.items():
            if key in a_lower:
                modalities.update(mods)
                break

    # Check for vision config keys
    for key in _VISION_KEYS:
        if key in config and config[key]:
            modalities.add("image")
            break

    # Check for audio config keys
    for key in _AUDIO_KEYS:
        if key in config and config[key]:
            modalities.add("audio")
            break

    # Check for video config keys
    for key in _VIDEO_KEYS:
        if key in config and config[key]:
            modalities.add("video")
            break

    return sorted(modalities), arch, model_type


def _get_hf_headers(token: str | None = None) -> dict:
    """Build auth headers for HuggingFace API."""
    t = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t:
        return {"Authorization": f"Bearer {t}"}
    return {}


def _fetch_from_api_metadata(model_id: str, timeout: int = 10) -> Optional[ModelInfo]:
    """Fallback: use HF API model metadata (works for gated models without auth)."""
    url = f"{HF_API}/{model_id}"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None

    tags = [t.lower() for t in data.get("tags", [])]
    pipeline = (data.get("pipeline_tag") or "").lower()

    modalities = set()
    # Check tags for multimodal hints
    vision_tags = ["vision", "image-text-to-text", "visual-question-answering", "image-to-text"]
    audio_tags = ["audio", "speech", "automatic-speech-recognition", "audio-to-text"]
    video_tags = ["video", "video-text-to-text"]

    for vt in vision_tags:
        if any(vt in t for t in tags) or vt in pipeline:
            modalities.add("image")
    for at in audio_tags:
        if any(at in t for t in tags) or at in pipeline:
            modalities.add("audio")
    for vdt in video_tags:
        if any(vdt in t for t in tags) or vdt in pipeline:
            modalities.add("video")

    if not modalities:
        return None

    return ModelInfo(
        name=model_id,
        multimodal=True,
        input_modalities=["text"] + sorted(modalities),
        output_modalities=["text"],
        source="huggingface-api",
        model_type=data.get("pipeline_tag"),
    )


def fetch_from_hub(model_id: str, token: str | None = None, timeout: int = 10) -> Optional[ModelInfo]:
    """Fetch model config from HuggingFace Hub and detect multimodal capabilities.

    For gated models, set HF_TOKEN env var or pass token directly.
    Falls back to HF API metadata if config.json is inaccessible.
    """
    url = HF_CONFIG_URL.format(model_id=model_id)
    headers = _get_hf_headers(token)

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code in (401, 403):
            logger.debug(f"Gated model {model_id}, trying API metadata fallback")
            return _fetch_from_api_metadata(model_id, timeout)
        if resp.status_code == 404:
            logger.debug(f"No config.json found for {model_id}")
            return None
        resp.raise_for_status()
        config = resp.json()
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch config for {model_id}: {e}")
        return _fetch_from_api_metadata(model_id, timeout)
    except ValueError:
        logger.warning(f"Invalid JSON in config for {model_id}")
        return None

    extra_modalities, arch, model_type = _detect_modalities_from_config(config)

    input_mods = ["text"] + extra_modalities
    is_multimodal = len(extra_modalities) > 0

    # Detect output modalities (most models are text-only output)
    output_mods = ["text"]

    return ModelInfo(
        name=model_id,
        multimodal=is_multimodal,
        input_modalities=input_mods,
        output_modalities=output_mods,
        architecture=arch,
        source="huggingface",
        model_type=model_type,
    )
