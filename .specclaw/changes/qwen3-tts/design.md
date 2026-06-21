# Design: qwen3-tts

## Technical Approach

Copy `outetts_tts.py`, change model name to `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`, adapter dir to `qwen3_tts_lora/`, keep `sample_rate=24000`. Add `qwen3` entry to `MODELS` dict in `synthesize.py`.

## File Changes Map

### `qwen3_tts.py` (new)
- Identical structure to `outetts_tts.py`
- `model_name = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"`
- `TTSSFTConfig(sample_rate=24000, ...)`
- Save to `qwen3_tts_lora/`

### `synthesize.py` (modify)
Add to MODELS dict:
```python
"qwen3": {
    "model_id": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
    "codec": None,
    "adapter": "qwen3_tts_lora",
},
```
Update `choices=list(MODELS)` — already dynamic, no explicit change needed.

### `README.md` (modify)
Add OuteTTS section after the OuteTTS section. Note multilingual capability (ZH/EN/JA/KO/+) and RAM (~5–6 GB bf16).

### `.gitignore` (modify)
Add `qwen3_tts_lora/` under the TTS adapter directories comment.
