"""vLLM supported multimodal models reference."""

from mmcheck.models import ModelInfo
from typing import Optional

# Models confirmed to work with vLLM's multimodal pipeline.
# Based on https://docs.vllm.ai/en/stable/models/supported_models.html
_VLLM_MULTIMODAL = {
    "LlavaForConditionalGeneration": ["image"],
    "LlavaNextForConditionalGeneration": ["image", "video"],
    "LlavaNextVideoForConditionalGeneration": ["video"],
    "LlavaOnevisionForConditionalGeneration": ["image", "video"],
    "MllamaForConditionalGeneration": ["image"],
    "Qwen2VLForConditionalGeneration": ["image", "video"],
    "Qwen2_5_VLForConditionalGeneration": ["image", "video"],
    "Qwen2AudioForConditionalGeneration": ["audio"],
    "Qwen2_5OmniModel": ["image", "audio", "video"],
    "InternVLChatModel": ["image"],
    "Phi3VForCausalLM": ["image"],
    "Phi4MMForCausalLM": ["image"],
    "PaliGemmaForConditionalGeneration": ["image"],
    "Gemma3ForConditionalGeneration": ["image"],
    "ChameleonForConditionalGeneration": ["image"],
    "FuyuForCausalLM": ["image"],
    "MiniCPMV": ["image", "video"],
    "MiniCPMO": ["image", "audio", "video"],
    "Idefics3ForConditionalGeneration": ["image"],
    "MolmoForCausalLM": ["image"],
    "AriaForConditionalGeneration": ["image"],
    "DeepseekVLV2ForCausalLM": ["image"],
    "PixtralForConditionalGeneration": ["image"],
    "GotOcr2ForConditionalGeneration": ["image"],
    "UltravoxModel": ["audio"],
    "WhisperForConditionalGeneration": ["audio"],
    "JanusForConditionalGeneration": ["image"],
    "Emu3ForConditionalGeneration": ["image"],
    "Florence2ForConditionalGeneration": ["image"],
}


def lookup_vllm(architecture: str) -> Optional[list[str]]:
    """Check if a model architecture is supported by vLLM for multimodal.

    Returns list of supported modalities or None.
    """
    return _VLLM_MULTIMODAL.get(architecture)


def is_vllm_multimodal(architecture: str) -> bool:
    """Check if a model architecture has vLLM multimodal support."""
    return architecture in _VLLM_MULTIMODAL


def enrich_with_vllm(info: ModelInfo) -> ModelInfo:
    """Add vLLM support info to an existing ModelInfo."""
    if info.architecture and info.architecture in _VLLM_MULTIMODAL:
        info.source = f"{info.source}+vllm"
    return info
