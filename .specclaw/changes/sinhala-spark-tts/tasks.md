# Tasks: Sinhala Spark-TTS Fine-Tuning

**Change:** sinhala-spark-tts
**Created:** 2026-06-20
**Total Tasks:** 4

## Summary

Create `sinhala_tts.py` and `synthesize.py`, add README section, then smoke-test end-to-end.

## Tasks

### Wave 1 — Implementation (parallel)

- [x] `T1` — Create sinhala_tts.py
  - Files: `sinhala_tts.py`
  - Estimate: medium
  - Depends: none
  - Notes: Mirror sinhala_asr.py structure. Use FastTTSModel.from_pretrained("mlx-community/Spark-TTS-0.5B-bf16"), FastTTSModel.get_peft_model(r=16, lora_alpha=16), TTSDataCollator(text_column="sentence", audio_column="audio_path"), TTSSFTTrainer with TTSSFTConfig(sample_rate=16000, max_steps=60, lr=2e-4). Support --tsv (default: data/sentences.tsv) and --resume flags. Resume: use load_weights(strict=False) not load_adapter(). Save to sinhala_tts_lora/.

- [x] `T2` — Create synthesize.py
  - Files: `synthesize.py`
  - Estimate: small
  - Depends: none
  - Notes: CLI: positional text arg, --adapter (default: sinhala_tts_lora), --out (default: output.wav). Load FastTTSModel, load adapter, call model.generate(text=...), save with soundfile. Create --out parent dir if needed. Exit(1) on empty text.

### Wave 2 — Documentation + Verification

- [x] `T3` — Add Sinhala TTS section to README.md
  - Files: `README.md`
  - Estimate: small
  - Depends: T1, T2
  - Notes: Add after the Sinhala ASR section. Cover: run command (uv run python sinhala_tts.py), expected training time (~3 min for 60 steps), synthesize command (python synthesize.py "ආයුබෝවන්" --out hello.wav), note that 50 samples is demo quality.

- [x] `T4` — Smoke-test sinhala_tts.py and synthesize.py
  - Files: none (read-only verification)
  - Estimate: medium
  - Depends: T1, T2, T3
  - Notes: Run `uv run python sinhala_tts.py --help` and `uv run python synthesize.py --help` to verify argparse. Run `uv run python -c "from mlx_tune import FastTTSModel, TTSSFTTrainer, TTSSFTConfig, TTSDataCollator; print('ok')"` to confirm imports. Full training test is T4 extended (manual, takes ~3 min).

---

## Legend

- `[ ]` Pending
- `[~]` In Progress
- `[x]` Complete
- `[!]` Failed
