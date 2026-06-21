# Proposal: qwen3-tts

**Status:** proposed
**Created:** 2026-06-20
**Part of ladder:** tts-model-ladder (step 2 of 3)
**Depends on:** outetts-tts (synthesize.py `--model` flag must exist)

---

## Problem Statement

After OuteTTS 1B is working, the next model on the ladder is Qwen3-TTS 1.7B — Alibaba's multilingual TTS model. It is notable for explicit support of Chinese, English, Japanese, Korean and more, making it the best candidate for Sinhala once enough training data is available. Its architecture is unique (dual embedding path, built-in 16-codebook speech tokenizer) but the training API is identical to the other models.

---

## Proposed Solution

Add Qwen3-TTS fine-tuning and extend `synthesize.py --model` with a `qwen3` option.

### Deliverables

1. **`qwen3_tts.py`** — fine-tuning script
   - `FastTTSModel.from_pretrained("mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16", max_seq_length=2048)`
   - No explicit codec kwarg (built-in 16-codebook speech tokenizer, auto-loaded)
   - `TTSSFTConfig(sample_rate=24000, max_steps=60, ...)`
   - Adapter saved to `qwen3_tts_lora/`
   - Flags: `--tsv` (default `data/sentences.tsv`), `--resume`

2. **`synthesize.py`** — add `qwen3` to the `--model` dispatch table

3. **README** — add Qwen3-TTS row, note multilingual capability and RAM (~5–6 GB)

4. **`.gitignore`** — add `qwen3_tts_lora/`

### RAM requirement

| Model | Quantisation | Approximate RAM |
|-------|-------------|-----------------|
| Qwen3-TTS 1.7B | bf16 | ~5–6 GB |

### Architecture note

Qwen3-TTS uses a dual embedding path (text and codec tokens use separate embeddings, summed) and a 5-layer code predictor that fills codebooks 1–15 after the 28-layer talker predicts codebook 0. The training API hides this complexity — `TTSDataCollator` and `TTSSFTTrainer` handle it transparently.

---

## Scope

### In
- `qwen3_tts.py`
- `synthesize.py` (add `qwen3` entry)
- `README.md`
- `.gitignore`

### Out
- Changes to `outetts_tts.py` or `sinhala_tts.py`
- Any orpheus code (handled in orpheus-tts change)

---

## Impact

| Area | Files | Complexity | Risk |
|------|-------|------------|------|
| New | `qwen3_tts.py` | low | low |
| Modify | `synthesize.py` | trivial (one dict entry) | none |
| Modify | `README.md`, `.gitignore` | trivial | none |

---

## Open Questions

None.
