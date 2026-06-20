# Design: Sinhala Spark-TTS Fine-Tuning

**Change:** sinhala-spark-tts
**Created:** 2026-06-20

## Technical Approach

Two new files mirroring the existing ASR pattern:

1. **`sinhala_tts.py`** — mirrors `sinhala_asr.py` but swaps `FastVisionModel` → `FastTTSModel`, `VLMSFTTrainer` → `TTSSFTTrainer`, and dataset format to `(text, audio)` pairs via `TTSDataCollator`.
2. **`synthesize.py`** — mirrors `transcribe.py` but calls `model.generate(text=...)` instead of audio input.

## Architecture

```
data/sentences.tsv (audio_path + sentence)
        ↓
  TTSDataCollator(text_column="sentence", audio_column="audio_path")
        ↓
  FastTTSModel (Spark-TTS-0.5B, Qwen2 backbone + BiCodec)
        ↓ LoRA r=16
  TTSSFTTrainer → sinhala_tts_lora/
        ↓
  synthesize.py: text → model.generate() → .wav
```

## File Changes Map

| File | Action | Description |
|------|--------|-------------|
| `sinhala_tts.py` | Create | TTS fine-tuning script |
| `synthesize.py` | Create | Text-to-speech inference CLI |
| `README.md` | Modify | Add Sinhala TTS section |

## Key Decisions

- **Reuse `data/sentences.tsv`** — columns map directly: `sentence` → `text_column`, `audio_path` → `audio_column`. No new dataset needed.
- **`--resume` uses `load_weights(strict=False)`** — same fix as `sinhala_asr.py`; `load_adapter()` freezes params and breaks training.
- **`TTSDataCollator` text/audio columns** — from the reference example: `text_column="text"`, `audio_column="audio"`. Our TSV uses `sentence` and `audio_path` — pass those column names directly to the collator.
- **60 steps, lr=2e-4** — matching reference example; suitable for 50-sample demo.
- **`synthesize.py` saves to file** — no real-time playback; user opens the `.wav`. Keeps deps minimal.

## API Reference

```python
# Fine-tuning
from mlx_tune import FastTTSModel, TTSSFTTrainer, TTSSFTConfig, TTSDataCollator

model, tokenizer = FastTTSModel.from_pretrained("mlx-community/Spark-TTS-0.5B-bf16")
model = FastTTSModel.get_peft_model(model, r=16, lora_alpha=16, target_modules=[...])

collator = TTSDataCollator(model, tokenizer, text_column="sentence", audio_column="audio_path")
trainer = TTSSFTTrainer(model, tokenizer, data_collator=collator, train_dataset=rows,
    args=TTSSFTConfig(sample_rate=16000, max_steps=60, ...))
trainer.train()
model.save_pretrained("sinhala_tts_lora")

# Inference
model, tokenizer = FastTTSModel.from_pretrained("mlx-community/Spark-TTS-0.5B-bf16")
model.load_adapter("sinhala_tts_lora")
FastTTSModel.for_inference(model)
audio = model.generate(text="ආයුබෝවන්", max_tokens=1024)
soundfile.write("output.wav", audio, 16000)
```

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `TTSDataCollator` column name mismatch | Medium | Verify at build time; adjust if needed |
| `model.generate()` TTS inference API differs from ASR | Medium | Check reference example; adapt |
| 50 samples produces low-quality synthesis | High | Expected for demo; document clearly in README |
