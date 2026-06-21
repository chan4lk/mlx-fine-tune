# Tasks: qwen3-tts

**Change:** qwen3-tts
**Created:** 2026-06-20
**Total Tasks:** 2

## Summary

Create `qwen3_tts.py`, add `qwen3` entry to `synthesize.py` MODELS dict, update README and `.gitignore`.

## Tasks

### Wave 1 — All tasks (parallel)

- [x] `T1` — Create qwen3_tts.py + extend synthesize.py
  - Files: `qwen3_tts.py`, `synthesize.py`
  - Estimate: small
  - Depends: none
  - Notes: Copy outetts_tts.py. Change model_name="mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16", adapter dir "qwen3_tts_lora". TTSSFTConfig(sample_rate=24000) unchanged. In synthesize.py, add "qwen3" entry to MODELS dict: model_id=above, codec=None, adapter="qwen3_tts_lora". The --model choices are already dynamic (list(MODELS)) so no further change needed.

- [x] `T2` — README + .gitignore
  - Files: `README.md`, `.gitignore`
  - Estimate: small
  - Depends: none
  - Notes: README: add Qwen3-TTS section after OuteTTS section. Include model name, sample rate (24 kHz), RAM (~5–6 GB bf16), multilingual note (ZH/EN/JA/KO and more), run command, synthesis example. .gitignore: add qwen3_tts_lora/ under TTS adapter directories comment.

---

## Legend

- `[ ]` Pending
- `[~]` In Progress
- `[x]` Complete
- `[!]` Failed
