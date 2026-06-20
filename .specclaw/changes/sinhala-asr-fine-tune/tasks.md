# Tasks: Sinhala ASR Fine-Tuning

**Change:** sinhala-asr-fine-tune
**Created:** 2026-06-18
**Total Tasks:** 4

## Summary

Create `sinhala_asr.py`, update `transcribe.py` with adapter flag, add README section, then smoke-test end-to-end (requires HF login).

## Tasks

### Wave 1 — Implementation (parallel)

- [x] `T1` — Create sinhala_asr.py
  - Files: `sinhala_asr.py`
  - Estimate: medium
  - Depends: none
  - Notes: Mirror audio_asr.py structure. Steps: (1) HF auth gate via HfApi().whoami(); (2) load Common Voice 17.0 si train[:200]; (3) resample audio to 16kHz WAV via librosa + soundfile into tempdir; (4) build messages dataset in VLMSFTTrainer format; (5) load Gemma 4 E4B with strict=False; (6) add LoRA; (7) train 50 steps; (8) save to sinhala_asr_lora/; (9) cleanup tempdir.

- [x] `T2` — Update transcribe.py with --adapter and --prompt flags
  - Files: `transcribe.py`
  - Estimate: small
  - Depends: none
  - Notes: Replace positional-only sys.argv with argparse. Keep backwards compatibility (audio path stays positional). Default adapter: gemma4_audio_asr_lora. Default prompt: "Transcribe this audio."

### Wave 2 — Documentation + Verification

- [x] `T3` — Add Sinhala ASR section to README.md
  - Files: `README.md`
  - Estimate: small
  - Depends: T1, T2
  - Notes: Cover: HF login requirement (`! hf auth login`), dataset acceptance URL, run command, expected training time (~2 min for 50 steps), how to transcribe with Sinhala adapter (`--adapter sinhala_asr_lora`).

- [x] `T4` — Smoke-test sinhala_asr.py and updated transcribe.py
  - Files: none (read-only verification)
  - Estimate: medium
  - Depends: T1, T2, T3
  - Notes: Requires HF login + Common Voice dataset access. Run `uv run python sinhala_asr.py` and confirm it reaches training. Then run `uv run python transcribe.py --help` to verify argparse. Full end-to-end transcription test is manual (requires actual Sinhala audio).

---

## Legend

- `[ ]` Pending
- `[~]` In Progress
- `[x]` Complete
- `[!]` Failed
