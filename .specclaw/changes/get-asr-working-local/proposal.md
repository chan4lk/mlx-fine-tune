# Proposal: Get ASR Working Locally

**Created:** 2026-06-17
**Status:** 🟡 Draft

## Problem

`audio_asr.py` exists in the project but has never been verified to run end-to-end on a local Apple Silicon machine. It depends on `soundfile` (not in `pyproject.toml`), uses synthetic audio for demo purposes, and requires the `mlx-community/gemma-4-e4b-it-4bit` model to be downloaded. There is no documented setup path — a developer cloning the repo cannot run ASR fine-tuning without trial-and-error.

## Proposed Solution

Make ASR fine-tuning runnable locally with a single command:

1. Add `soundfile` to the project dependencies in `pyproject.toml`.
2. Verify `audio_asr.py` runs end-to-end on Apple Silicon (model download, synthetic audio generation, training loop, inference, save).
3. Fix any runtime errors encountered during the local run.
4. Document the ASR setup and run instructions in `README.md`.

## Scope

### In Scope
- Add `soundfile` dependency to `pyproject.toml`
- End-to-end smoke test of `audio_asr.py` on local machine
- Fix any import errors, model-loading issues, or API mismatches
- Update README with ASR-specific instructions

### Out of Scope
- Replacing synthetic audio with real datasets (Common Voice, LibriSpeech)
- Hyperparameter tuning or model quality improvements
- CI/CD integration

## Impact

- **Files affected:** 3 (estimated) — `pyproject.toml`, `audio_asr.py`, `README.md`
- **Complexity:** small
- **Risk:** low — changes are additive; no existing functionality is modified

## Open Questions

1. Does `mlx-tune[audio]` already bundle `soundfile`, or is it a separate install?
2. Are there any MLX version constraints for Gemma 4's audio tower?
3. Should the README document a minimal-memory path (E2B vs E4B)?

---

**To proceed:** Review this proposal and approve to begin planning.
