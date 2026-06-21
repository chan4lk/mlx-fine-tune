# Proposal: orpheus-tts

**Status:** proposed
**Created:** 2026-06-20
**Part of ladder:** tts-model-ladder (step 3 of 3)
**Depends on:** outetts-tts (synthesize.py `--model` flag must exist)

---

## Problem Statement

Orpheus 3B is the largest and highest-quality TTS model available via mlx-tune. It uses a SNAC 24 kHz codec (which must be named explicitly in `from_pretrained`) and is Llama-based. On Apple Silicon with 16 GB+ RAM it should produce the best audio quality of the four models in the ladder.

---

## Proposed Solution

Add Orpheus fine-tuning and extend `synthesize.py --model` with an `orpheus` option.

### Deliverables

1. **`orpheus_tts.py`** — fine-tuning script
   - `FastTTSModel.from_pretrained("mlx-community/orpheus-3b-0.1-ft-bf16", max_seq_length=2048, codec_model="mlx-community/snac_24khz")`
   - SNAC codec must be specified explicitly (unlike OuteTTS/Qwen3/Spark)
   - `TTSSFTConfig(sample_rate=24000, max_steps=60, ...)`
   - Adapter saved to `orpheus_lora/`
   - Flags: `--tsv` (default `data/sentences.tsv`), `--resume`

2. **`synthesize.py`** — add `orpheus` to the `--model` dispatch table (includes `codec_model`)

3. **README** — add Orpheus row, note 16 GB RAM recommendation and highest quality ceiling

4. **`.gitignore`** — add `orpheus_lora/`

### RAM requirement

| Model | Quantisation | Approximate RAM |
|-------|-------------|-----------------|
| Orpheus 3B | bf16 | ~6–8 GB |
| Recommended | — | 16 GB+ |

### Key difference from other models

`codec_model` must be passed to `from_pretrained` — omitting it causes a runtime error. The dispatch table in `synthesize.py` carries this as an optional field so the pattern stays uniform.

---

## Scope

### In
- `orpheus_tts.py`
- `synthesize.py` (add `orpheus` entry with `codec_model`)
- `README.md`
- `.gitignore`

### Out
- Changes to other TTS scripts
- Voice cloning / reference-audio conditioning (future)

---

## Impact

| Area | Files | Complexity | Risk |
|------|-------|------------|------|
| New | `orpheus_tts.py` | low | low |
| Modify | `synthesize.py` | trivial (one dict entry) | none |
| Modify | `README.md`, `.gitignore` | trivial | none |

---

## Open Questions

None.
