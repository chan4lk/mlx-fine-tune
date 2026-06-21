# Design: orpheus-tts

## Technical Approach

Copy `qwen3_tts.py`, change model name and add `codec_model` kwarg. Add `orpheus` entry to `synthesize.py` MODELS dict — the `codec` field is non-None, which triggers the existing `from_pretrained_kwargs["codec_model"] = cfg["codec"]` path.

## File Changes Map

### `orpheus_tts.py` (new)
- Identical structure to `qwen3_tts.py`
- `model_name = "mlx-community/orpheus-3b-0.1-ft-bf16"`
- `codec_model = "mlx-community/snac_24khz"` passed to `from_pretrained`
- `TTSSFTConfig(sample_rate=24000, ...)`
- Save to `orpheus_lora/`

### `synthesize.py` (modify)
Add to MODELS dict:
```python
"orpheus": {
    "model_id": "mlx-community/orpheus-3b-0.1-ft-bf16",
    "codec": "mlx-community/snac_24khz",
    "adapter": "orpheus_lora",
},
```
The existing `if cfg["codec"]: from_pretrained_kwargs["codec_model"] = cfg["codec"]` path handles this correctly.

### `README.md` (modify)
Add Orpheus section after Qwen3-TTS section. Note: largest model (3B), highest quality ceiling, 16 GB RAM recommended, ~6–8 GB minimum.

### `.gitignore` (modify)
Add `orpheus_lora/` under TTS adapter directories comment.
