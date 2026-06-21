# Tasks: outetts-tts

**Change:** outetts-tts
**Created:** 2026-06-20
**Total Tasks:** 3

## Summary

Create `outetts_tts.py`, update `synthesize.py` with `--model` flag + `model.sample_rate`, update README and `.gitignore`.

## Tasks

### Wave 1 — Implementation (parallel)

- [x] `T1` — Create outetts_tts.py
  - Files: `outetts_tts.py`
  - Estimate: small
  - Depends: none
  - Notes: Copy sinhala_tts.py structure exactly. Change: model_name="mlx-community/Llama-OuteTTS-1.0-1B-8bit", no codec_model kwarg, TTSSFTConfig(sample_rate=24000), save to outetts_lora/. Keep load_samples(), --tsv, --resume (load_weights strict=False) unchanged.

- [x] `T2` — Update synthesize.py: --model flag + model.sample_rate
  - Files: `synthesize.py`
  - Estimate: small
  - Depends: none
  - Notes: Add MODELS dict at top: spark → {model_id, codec=None, adapter="sinhala_tts_lora"}, outetts → {model_id, codec=None, adapter="outetts_lora"}. Replace --adapter with --model (choices=["spark","outetts"], default="spark"). Derive adapter from MODELS[args.model]["adapter"]. Pass codec_model only if MODELS[args.model]["codec"] is not None. Replace sf.write(..., 16000) with sf.write(..., model.sample_rate). Print "Model: {args.model}" instead of adapter path.

### Wave 2 — Docs + housekeeping

- [x] `T3` — README + .gitignore
  - Files: `README.md`, `.gitignore`
  - Estimate: small
  - Depends: T1, T2
  - Notes: README: add OuteTTS section after Spark-TTS section. Include: run command, RAM note (~3–4 GB, 8-bit quant), synthesis example with --model outetts. .gitignore: add outetts_lora/ entry near sinhala_tts_lora (which is not explicitly listed but *.safetensors covers weights — add the dir itself for clarity).

---

## Legend

- `[ ]` Pending
- `[~]` In Progress
- `[x]` Complete
- `[!]` Failed
