"""Tests for mmcheck."""

from mmcheck import check
from mmcheck.models import ModelInfo
from mmcheck.registry import lookup


def test_registry_gpt4o():
    info = lookup("gpt-4o")
    assert info is not None
    assert info.multimodal is True
    assert "image" in info.input_modalities
    assert "audio" in info.input_modalities
    assert info.source == "registry"


def test_registry_text_only():
    info = lookup("gpt-4")
    assert info is not None
    assert info.multimodal is False
    assert info.input_modalities == ["text"]


def test_registry_claude():
    info = lookup("claude-3-5-sonnet")
    assert info is not None
    assert info.multimodal is True
    assert "image" in info.input_modalities
    assert "audio" not in info.input_modalities


def test_registry_gemini():
    info = lookup("gemini-2.5-pro")
    assert info is not None
    assert info.multimodal is True
    assert "video" in info.input_modalities


def test_registry_not_found():
    info = lookup("nonexistent-model-xyz")
    assert info is None


def test_registry_prefix_strip():
    info = lookup("openai/gpt-4o")
    assert info is not None
    assert info.multimodal is True


def test_model_info_supports():
    info = ModelInfo(
        name="test",
        multimodal=True,
        input_modalities=["text", "image"],
        output_modalities=["text"],
    )
    assert info.supports("image") is True
    assert info.supports("Image") is True  # case insensitive
    assert info.supports("audio") is False


def test_model_info_str():
    info = ModelInfo(name="test-model", multimodal=True, input_modalities=["text", "image"])
    s = str(info)
    assert "test-model" in s
    assert "Yes" in s


def test_check_registry_hit():
    info = check("gpt-4o", offline=True)
    assert info.multimodal is True
    assert info.source == "registry"


def test_check_unknown_offline():
    info = check("totally-unknown-model", offline=True)
    assert info.multimodal is False
    assert info.source == "unknown"


def test_check_hf_gemma3():
    """Integration test - requires network."""
    info = check("google/gemma-3-4b-it")
    assert info is not None
    assert info.multimodal is True
    assert "image" in info.input_modalities


def test_check_hf_text_only():
    """Integration test - requires network."""
    info = check("google/gemma-2-2b-it")
    assert info is not None
    assert info.multimodal is False
