# Tasks: orpheus-tts

**Change:** orpheus-tts
**Created:** 2026-06-20
**Total Tasks:** 2

## Summary

Create `orpheus_tts.py`, add `orpheus` entry to `synthesize.py` MODELS dict (with non-None codec), update README and `.gitignore`.

## Tasks

### Wave 1 — All tasks (parallel)

- [x] `T1` — Create orpheus_tts.py + extend synthesize.py
  - Files: `orpheus_tts.py`, `synthesize.py`
  - Estimate: small
  - Depends: none
  - Notes: Copy qwen3_tts.py. Change model_name="mlx-community/orpheus-3b-0.1-ft-bf16". Add codec_model="mlx-community/snac_24khz" to from_pretrained call. adapter dir "orpheus_lora". TTSSFTConfig(sample_rate=24000) unchanged. In synthesize.py add "orpheus" to MODELS dict: model_id=above, codec="mlx-community/snac_24khz", adapter="orpheus_lora". The existing codec path in synthesize.py handles non-None codec already.

- [x] `T2` — README + .gitignore
  - Files: `README.md`, `.gitignore`
  - Estimate: small
  - Depends: none
  - Notes: README: add Orpheus section after Qwen3-TTS section. Include model name, size (3B), sample rate (24 kHz), RAM note (min ~6–8 GB, 16 GB recommended for best results), quality note (highest ceiling in the ladder), SNAC codec note, run command, synthesis example. .gitignore: add orpheus_lora/ under TTS adapter directories comment.

---

## Legend

- `[ ]` Pending
- `[~]` In Progress
- `[x]` Complete
- `[!]` Failed
