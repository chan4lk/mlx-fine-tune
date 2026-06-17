# Tasks: Get ASR Working Locally

**Change:** get-asr-working-local
**Created:** 2026-06-17
**Total Tasks:** 3

## Summary

Two code edits (usage comment + README) followed by a local smoke-test. No new dependencies or files.

## Tasks

### Wave 1 — Code fixes

- [x] `T1` — Fix usage comment in audio_asr.py
  - Files: `audio_asr.py` (line 19)
  - Estimate: small
  - Depends: none
  - Notes: Change `python examples/47_gemma4_audio_asr_finetuning.py` → `python audio_asr.py`

- [x] `T2` — Add ASR setup section to README.md
  - Files: `README.md`
  - Estimate: small
  - Depends: none
  - Notes: Add under the existing "Examples" section. Cover: model size (~2 GB), memory requirement (8 GB RAM minimum), run command (`uv run python audio_asr.py`), E2B vs E4B trade-off, pointer to real datasets (Common Voice, LibriSpeech, FLEURS), note that Apple Silicon is required.

### Wave 2 — Verification

- [x] `T3` — Smoke-test end-to-end run
  - Files: none (read-only verification)
  - Estimate: medium
  - Depends: T1, T2
  - Notes: Run `uv run python audio_asr.py` and confirm all 7 steps complete and output ends with "Done! Gemma 4 audio ASR fine-tuning complete." Model download (~2 GB) required on first run.

---

## Legend

- `[ ]` Pending
- `[~]` In Progress
- `[x]` Complete
- `[!]` Failed
