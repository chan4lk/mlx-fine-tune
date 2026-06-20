# Proposal: Sinhala ASR Fine-Tuning

**Created:** 2026-06-17
**Status:** 🟡 Draft

## Problem

The existing `audio_asr.py` was trained on 10 synthetic English audio clips — far too few and wrong language to transcribe Sinhala speech. Attempting to transcribe a Sinhala song produced hallucinated English text about MLX. We need a Sinhala-specific fine-tune of Gemma 4's audio tower using real, validated Sinhala speech data.

## Proposed Solution

Create `sinhala_asr.py` — a new fine-tuning script that:

1. Loads the Mozilla Common Voice 17.0 Sinhala dataset (`mozilla-foundation/common_voice_17_0`, language `"si"`) — validated clips with ground-truth Sinhala transcriptions.
2. Converts the MP3 audio to 16kHz WAV (required by Gemma 4's audio tower).
3. Fine-tunes Gemma 4 E4B with LoRA on (audio → Sinhala transcription) pairs.
4. Saves adapters to `sinhala_asr_lora/`.
5. Updates `transcribe.py` to accept an optional `--adapter` flag so the same script can be used with any adapter set.

## Scope

### In Scope
- New script `sinhala_asr.py` for Sinhala fine-tuning
- Audio preprocessing: MP3 → 16kHz WAV conversion via `soundfile` + `librosa`
- Dataset loader for Common Voice 17.0 Sinhala (train split, capped at a reasonable sample count)
- LoRA fine-tune of Gemma 4 E4B with same config as `audio_asr.py`
- Update `transcribe.py` to accept `--adapter <path>` argument
- README update: Sinhala ASR section

### Out of Scope
- The local Mozilla SPS corpus (no transcriptions available)
- Evaluation metrics (WER/CER) — out of scope for this iteration
- Deploying or serving the model
- Training on GPU / cloud hardware

## Impact

- **Files affected:** 3 new + 2 modified (`sinhala_asr.py`, `transcribe.py`, `README.md`, `audio_asr.py` no change, `.specclaw/`)
- **Complexity:** medium
- **Risk:** low — additive only; no existing scripts modified

## Open Questions

1. How many validated Sinhala clips does Common Voice 17.0 have? (Need ≥ 100 for meaningful fine-tuning.)
2. Does `mlx-tune[audio]` include `librosa` for MP3→WAV conversion, or do we need to add it?
3. Should we cap training samples (e.g. 500 max) to keep the demo fast, or use all validated clips?

---

**To proceed:** Review this proposal and approve to begin planning.
