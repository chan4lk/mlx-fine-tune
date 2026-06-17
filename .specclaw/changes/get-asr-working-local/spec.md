# Spec: Get ASR Working Locally

**Change:** get-asr-working-local
**Created:** 2026-06-17
**Status:** 🟢 Approved

## Overview

Make `audio_asr.py` runnable end-to-end on a local Apple Silicon machine with a single documented command. All dependencies are already present via `mlx-tune[audio]`; the gaps are a wrong usage comment, missing README instructions, and no documented model download step.

## Requirements

### Functional Requirements

- FR1: `uv run python audio_asr.py` completes without unhandled exceptions on Apple Silicon.
- FR2: The usage comment in `audio_asr.py` reflects the correct invocation path (`python audio_asr.py`, not `python examples/47_...`).
- FR3: `README.md` documents the ASR setup: model size (~2 GB download), memory requirement, and the run command.
- FR4: `README.md` documents how to swap in real audio datasets (Common Voice, LibriSpeech) in place of the synthetic demo.

### Non-Functional Requirements

- NFR1: No new runtime dependencies beyond `mlx-tune[audio]` — `soundfile` is already bundled transitively and must not be added as a top-level dep.
- NFR2: Script must work on both E2B (~1 GB) and E4B (~2 GB) variants; README notes the difference.

## Acceptance Criteria

- AC1: Running `uv run python audio_asr.py` on an M-series Mac completes all 7 steps and prints "Done! Gemma 4 audio ASR fine-tuning complete."
- AC2: The usage comment at the top of `audio_asr.py` reads `python audio_asr.py`.
- AC3: `README.md` has an ASR section listing model size, memory requirement, and the run command.
- AC4: `README.md` mentions at least one real dataset (e.g. Common Voice) as a production replacement.
- AC5: `pyproject.toml` is unchanged — `soundfile` is NOT added as an explicit dependency.

## Edge Cases

- EC1: Model download fails (no internet / disk full) — script will surface the HuggingFace error; no special handling required.
- EC2: Pre/post-training inference steps raise an exception — already wrapped in `try/except` in the script; training must still complete.
- EC3: Running on an Intel Mac — not supported (MLX requires Apple Silicon); README should note this.

## Dependencies

- `mlx-tune[audio] >= 0.5.1` (already in `pyproject.toml`)
- `mlx-community/gemma-4-e4b-it-4bit` model (~2 GB, downloaded on first run via HuggingFace)

## Notes

- `soundfile 0.14.0` is already installed transitively via `librosa` ← `mlx-tune[audio]`. No explicit dep needed.
- All `FastVisionModel`, `VLMSFTTrainer`, `VLMSFTConfig`, and `VLMModelWrapper.generate` APIs confirmed present in `mlx-tune 0.5.1`.
