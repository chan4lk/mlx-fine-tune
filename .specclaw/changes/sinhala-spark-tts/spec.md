# Spec: Sinhala Spark-TTS Fine-Tuning

**Change:** sinhala-spark-tts
**Created:** 2026-06-20
**Status:** 🟢 Approved

## Overview

Fine-tune Spark-TTS (`mlx-community/Spark-TTS-0.5B-bf16`) on the existing Sinhala voice dataset (`data/sentences.tsv`) to produce a model that synthesizes Sinhala speech from text. Adds `sinhala_tts.py` (fine-tuning) and `synthesize.py` (inference).

## Functional Requirements

- FR1: `sinhala_tts.py` loads `data/sentences.tsv` (or any `--tsv` path) and fine-tunes Spark-TTS using `FastTTSModel` + `TTSSFTTrainer`.
- FR2: The script accepts `--tsv <path>` to use any TSV with `audio_path` and `sentence` columns.
- FR3: The script accepts `--resume <adapter_dir>` to continue training from an existing adapter (same pattern as `sinhala_asr.py`).
- FR4: Adapters are saved to `sinhala_tts_lora/` on completion.
- FR5: `synthesize.py` accepts a Sinhala text string and generates a `.wav` file.
- FR6: `synthesize.py` accepts `--adapter <path>` (default: `sinhala_tts_lora`) and `--out <file>` (default: `output.wav`).
- FR7: README documents the Sinhala TTS section: run command, expected training time, how to synthesize.

## Non-Functional Requirements

- NFR1: No new top-level dependencies — `FastTTSModel`, `TTSSFTTrainer`, `TTSSFTConfig`, `TTSDataCollator` are already available via `mlx-tune[audio]`.
- NFR2: `sinhala_tts.py` mirrors the structure of `sinhala_asr.py` for consistency.
- NFR3: Script must run end-to-end on Apple Silicon with `Spark-TTS-0.5B-bf16` (~1GB model).

## Acceptance Criteria

- AC1: `uv run python sinhala_tts.py` loads `data/sentences.tsv`, trains for up to 60 steps, and saves `sinhala_tts_lora/adapters.safetensors`.
- AC2: `uv run python sinhala_tts.py --tsv recordings/transcriptions-001.tsv` works with the recordings TSV.
- AC3: `uv run python sinhala_tts.py --resume sinhala_tts_lora` continues training without "no trainable parameters" error (same `load_weights` fix as ASR).
- AC4: `uv run python synthesize.py "ආයුබෝවන්"` produces `output.wav`.
- AC5: `uv run python synthesize.py --help` shows `--adapter`, `--out`, and text positional arg.
- AC6: `pyproject.toml` is unchanged.
- AC7: README has a Sinhala TTS section with run and synthesize commands.

## Edge Cases

- EC1: TSV has fewer than 10 samples → train on all available; log the count.
- EC2: `--out` directory doesn't exist → create it automatically.
- EC3: `--resume` adapter not found → print clear error and exit(1).
- EC4: Text input is empty string → print error and exit(1).
