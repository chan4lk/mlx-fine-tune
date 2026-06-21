# Proposal: outetts-tts

**Status:** proposed
**Created:** 2026-06-20
**Part of ladder:** tts-model-ladder (step 1 of 3)

---

## Problem Statement

Spark-TTS 0.5B is the only fine-tunable TTS model in the project. OuteTTS 1B is the next step up — a Llama-based model with a DAC codec that produces 24 kHz audio and has been shown to generalise well to low-resource languages with small datasets.

---

## Proposed Solution

Add OuteTTS fine-tuning and inference support, and introduce the `--model` flag in `synthesize.py` so all TTS models share one inference entry point.

### Deliverables

1. **`outetts_tts.py`** — fine-tuning script
   - `FastTTSModel.from_pretrained("mlx-community/Llama-OuteTTS-1.0-1B-8bit", max_seq_length=2048)`
   - No explicit codec kwarg (DAC auto-detected from the outetts model profile)
   - `TTSSFTConfig(sample_rate=24000, max_steps=60, ...)`
   - Adapter saved to `outetts_lora/`
   - Flags: `--tsv` (default `data/sentences.tsv`), `--resume`
   - Loads WAV arrays via soundfile (same pattern as `sinhala_tts.py`)

2. **`synthesize.py` — add `--model` flag**
   - Choices: `spark` (default, existing behaviour) | `outetts`
   - Dispatch table maps model name → `(model_id, codec_model_or_None, adapter_dir)`
   - Use `model.sample_rate` when writing WAV (replaces hardcoded `16000`)
   - Subsequent TTS changes (qwen3-tts, orpheus-tts) add entries to this table

3. **README** — add OuteTTS row to Sinhala TTS section, note RAM requirement (~3–4 GB)

4. **`.gitignore`** — add `outetts_lora/`

### RAM requirement

| Model | Quantisation | Approximate RAM |
|-------|-------------|-----------------|
| OuteTTS 1B | 8-bit | ~3–4 GB |

---

## Scope

### In
- `outetts_tts.py`
- `synthesize.py` (add `--model` flag, fix sample rate)
- `README.md`
- `.gitignore`

### Out
- `record_dataset.py`, `speak.py` (unchanged)
- Voice cloning / reference audio
- Any qwen3 or orpheus code (handled in later changes)

---

## Impact

| Area | Files | Complexity | Risk |
|------|-------|------------|------|
| New | `outetts_tts.py` | low | low |
| Modify | `synthesize.py` | low | low — `spark` default preserves existing behaviour |
| Modify | `README.md`, `.gitignore` | trivial | none |

---

## Open Questions

None — design decisions resolved in tts-model-ladder proposal review.
