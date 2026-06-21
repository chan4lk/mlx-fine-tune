# Design: outetts-tts

## Technical Approach

Mechanically adapt `sinhala_tts.py` for OuteTTS. The only differences are:
- `model_name` string
- `sample_rate=24000` in TTSSFTConfig (OuteTTS is 24 kHz)
- adapter output dir `outetts_lora/`

For `synthesize.py`, introduce a `MODELS` dict and a `--model` flag. The dict holds everything needed to load each model; the rest of the script is unchanged.

## File Changes Map

### `outetts_tts.py` (new)

```
from_pretrained("mlx-community/Llama-OuteTTS-1.0-1B-8bit", max_seq_length=2048)
get_peft_model(r=16, lora_alpha=16, target_modules=[q/k/v/o/gate/up/down_proj])
--tsv default=data/sentences.tsv
--resume → load_weights(strict=False)  [same fix as sinhala_tts.py]
TTSSFTConfig(sample_rate=24000, max_steps=60, lr=2e-4, train_on_completions=True)
save_pretrained("outetts_lora/")
```

### `synthesize.py` (modify)

Add at module level:
```python
MODELS = {
    "spark":   {"model_id": "mlx-community/Spark-TTS-0.5B-bf16",      "codec": None,                           "adapter": "sinhala_tts_lora"},
    "outetts": {"model_id": "mlx-community/Llama-OuteTTS-1.0-1B-8bit", "codec": None,                           "adapter": "outetts_lora"},
}
```

Replace the `--adapter` flag with `--model {spark,outetts}` (default: `spark`).
Derive `adapter_dir = MODELS[args.model]["adapter"]`.
Pass `codec_model=MODELS[args.model]["codec"]` to `from_pretrained` (skip kwarg if None).
Replace `sf.write(wav_path, audio, 16000)` → `sf.write(wav_path, audio, model.sample_rate)`.

**Backward-compatible:** `spark` default → same model, same adapter dir.

### `README.md` (modify)

Add under Sinhala TTS section:
```
#### OuteTTS 1B (24 kHz)
RAM: ~3–4 GB  
uv run python outetts_tts.py [--tsv PATH] [--resume outetts_lora]
uv run python synthesize.py "ආයුබෝවන්" --model outetts
```

### `.gitignore` (modify)

Add `outetts_lora/` alongside existing adapter-dir entries.

## Key Decisions

1. **`--model` replaces `--adapter`** — more ergonomic; the adapter dir is derived from the model name. Users who need a non-default adapter dir can be addressed in a future flag if needed.
2. **`codec` field is None for spark/outetts** — Orpheus is the only model needing an explicit codec; keeping the field in the dict makes the pattern uniform across all future entries.
3. **`model.sample_rate`** — avoids hardcoding; correct for all current and future models.

## Risks

- OuteTTS DAC codec auto-detection depends on the model profile tag in the HF repo — if a future model version changes this, `from_pretrained` may fail. Mitigation: document the explicit HF model ID in the script.
