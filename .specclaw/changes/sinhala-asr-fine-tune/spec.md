# Spec: Sinhala ASR Fine-Tuning

**Change:** sinhala-asr-fine-tune
**Created:** 2026-06-18
**Status:** 🟢 Approved

## Overview

Fine-tune Gemma 4 E4B for Sinhala ASR using validated Mozilla Common Voice 17.0 Sinhala clips. Produces `sinhala_asr_lora/` adapters and updates `transcribe.py` to support adapter selection.

## Requirements

### Functional Requirements

- FR1: `sinhala_asr.py` loads Mozilla Common Voice 17.0 Sinhala (`"si"`) train split via the HuggingFace `datasets` library.
- FR2: Audio is resampled to 16kHz mono WAV in memory using `librosa.resample` before being passed to the trainer (Common Voice clips are MP3, often at 48kHz).
- FR3: The script fine-tunes Gemma 4 E4B with LoRA on (audio → Sinhala `sentence`) pairs, capping at 200 samples for a manageable demo run.
- FR4: Adapters are saved to `sinhala_asr_lora/`.
- FR5: `transcribe.py` accepts `--adapter <path>` (default: `gemma4_audio_asr_lora`) and `--prompt <text>` (default: `"Transcribe this audio."`) CLI arguments.
- FR6: A HuggingFace login gate is added: if the user is not authenticated, the script prints a clear message and exits with a non-zero code (`hf auth login` required for gated Common Voice dataset).
- FR7: README documents the Sinhala ASR section: HF login requirement, run command, expected training time, and how to transcribe with the Sinhala adapter.

### Non-Functional Requirements

- NFR1: No new top-level dependencies in `pyproject.toml` — `librosa` and `datasets` are already bundled transitively via `mlx-tune[audio]`.
- NFR2: `strict=False` must be passed to `from_pretrained` (same as `audio_asr.py`) for the mlx_vlm 0.6.3 bug.
- NFR3: Script must run end-to-end on Apple Silicon with the model already cached.

## Acceptance Criteria

- AC1: `uv run python sinhala_asr.py` loads Common Voice Sinhala, prints sample count, trains for up to 50 steps, and saves `sinhala_asr_lora/adapters.safetensors`.
- AC2: `uv run python transcribe.py <audio> --adapter sinhala_asr_lora` runs inference with the Sinhala adapters.
- AC3: `uv run python transcribe.py --help` shows both `--adapter` and `--prompt` flags.
- AC4: If the user is not logged in to HuggingFace, the script exits with a clear error message (not a cryptic stack trace).
- AC5: `pyproject.toml` is unchanged.
- AC6: README has a Sinhala ASR section with login and run instructions.

## Edge Cases

- EC1: HuggingFace not logged in → AC4 (explicit error, no crash).
- EC2: Common Voice Sinhala train split has fewer than 200 clips → use all available.
- EC3: Audio already at 16kHz → `librosa.resample` is a no-op; no double-resampling.
- EC4: Audio saved as temp file → clean up after training.

## Dependencies

- `mozilla-foundation/common_voice_17_0` (gated — requires HF login + dataset agreement)
- `librosa` (already in `mlx-tune[audio]` transitively)
- `mlx-tune[audio] >= 0.5.1` (already in `pyproject.toml`)

## Notes

- Common Voice 17.0 requires HF auth; user must run `! hf auth login` once.
- Dataset column: audio is in `audio["array"]` (numpy) + `audio["sampling_rate"]`; transcription is `sentence`.
- Pattern from `audio_asr.py`: messages format `[{role: user, content: [{type: audio, audio: path}, {type: text, text: prompt}]}, {role: assistant, content: [{type: text, text: transcription}]}]`.
- Temp WAV files written to `tempfile.mkdtemp()` and cleaned up after training.
