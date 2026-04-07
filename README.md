# mmcheck

Check if a model supports multimodal inputs.

```bash
pip install mmcheck
```

## Quick start

```python
from mmcheck import check

info = check("google/gemma-4-27b-it")
info.multimodal        # True
info.input_modalities  # ["text", "image", "video"]
info.supports("image") # True
info.supports("audio") # False
```

## CLI

```bash
mmcheck google/gemma-4-27b-it
# Model:      google/gemma-4-27b-it
# Multimodal: YES
# Inputs:     text, image, video
# Outputs:    text

mmcheck meta-llama/Llama-3-8B
# Multimodal: NO

mmcheck --json google/gemma-4-27b-it
mmcheck --offline gemma-4-27b-it
```

## How it works

Three layers, checked in order:

1. **Built-in registry** — 30+ popular models (GPT-4o, Claude, Gemini, Llama, Qwen). Instant, no network.
2. **HuggingFace Hub** — fetches `config.json`, looks for `vision_config`, `audio_encoder`, architecture class names.
3. **vLLM cross-reference** — tags models with vLLM multimodal support status.

| Modality | Detection |
|----------|-----------|
| Image | `vision_config`, `vision_tower`, known VLM architectures |
| Audio | `audio_config`, `audio_encoder`, Whisper, Ultravox |
| Video | `video_config`, LLaVA-Next-Video, MiniCPM-V |

## License

MIT
