# Proposal: Sinhala Spark-TTS Fine-Tuning

**Change:** sinhala-spark-tts
**Created:** 2026-06-20
**Status:** 🟡 Pending Approval

## Problem Statement

The project can transcribe Sinhala speech → text (ASR) but cannot go in the reverse direction: text → Sinhala speech (TTS). A Sinhala TTS model would enable synthesizing new audio from any Sinhala sentence — useful for voice assistants, accessibility tools, and generating more training data for the ASR model.

## Proposed Solution

Fine-tune **Spark-TTS** (`mlx-community/Spark-TTS-0.5B-bf16`) on the Sinhala voice recordings already in `data/sentences.tsv` using `FastTTSModel` + `TTSSFTTrainer` from mlx-tune. Spark-TTS is a 0.5B Qwen2-based model with a BiCodec audio tokenizer — it runs well on Apple Silicon and the existing dataset format is directly compatible.

Two new files:

| File | Purpose |
|------|---------|
| `sinhala_tts.py` | Fine-tune Spark-TTS on `data/sentences.tsv` (or any `--tsv`); save adapters to `sinhala_tts_lora/` |
| `synthesize.py` | CLI: given a Sinhala text string, generate a `.wav` file using the fine-tuned adapter |

## Key API (from reference example)

```python
from mlx_tune import FastTTSModel, TTSSFTTrainer, TTSSFTConfig, TTSDataCollator

model, tokenizer = FastTTSModel.from_pretrained("mlx-community/Spark-TTS-0.5B-bf16")
model = FastTTSModel.get_peft_model(model, r=16, lora_alpha=16, target_modules=[...])

collator = TTSDataCollator(model, tokenizer, text_column="text", audio_column="audio")
trainer = TTSSFTTrainer(model, tokenizer, data_collator=collator, args=TTSSFTConfig(...))
trainer.train()
```

The existing `data/sentences.tsv` (`audio_path`, `sentence` columns) maps directly to `audio` + `text` — no new dataset needed.

## Scope

**In:**
- `sinhala_tts.py` — fine-tuning script with `--tsv`, `--resume`, `--steps` flags (mirrors `sinhala_asr.py` patterns)
- `synthesize.py` — inference: `python synthesize.py "ආයුබෝවන්" --adapter sinhala_tts_lora --out hello.wav`
- Update `README.md` with Sinhala TTS section

**Out:**
- Real-time audio playback (scope creep; user can open the `.wav`)
- Multi-speaker voice cloning
- New dataset recording (existing `data/` is sufficient for a demo run)

## Impact

| Area | Detail |
|------|--------|
| New files | `sinhala_tts.py`, `synthesize.py` |
| Modified files | `README.md` |
| New model | `mlx-community/Spark-TTS-0.5B-bf16` (~1GB download, cached after first run) |
| New adapter output | `sinhala_tts_lora/` |
| Risk | Low — mirrors existing ASR pattern; `FastTTSModel` API is stable |

## Open Questions

1. Does `mlx-tune` installed in this project already include `FastTTSModel` / `TTSSFTTrainer`? (Likely yes via `mlx-tune[audio]` — needs a quick import check at build time.)
2. 50 training samples is minimal for TTS quality. Is the goal a working demo or production-quality voice? (Proposal assumes demo.)
3. Should `synthesize.py` also support `--no-adapter` to compare base model vs fine-tuned output?
