# Design: Sinhala ASR Fine-Tuning

**Change:** sinhala-asr-fine-tune
**Created:** 2026-06-18

## Technical Approach

Two new files + one update to `transcribe.py`:

1. **`sinhala_asr.py`** — mirrors `audio_asr.py` structure but replaces synthetic audio generation with a Common Voice loader + `librosa` resampler.
2. **`transcribe.py`** — add `argparse` with `--adapter` and `--prompt` flags.
3. **`README.md`** — add Sinhala ASR section.

## Architecture

```
Common Voice 17.0 (HF, gated)
        ↓
  datasets.load_dataset("si", split="train[:200]")
        ↓ audio["array"] + audio["sampling_rate"]
  librosa.resample(→ 16kHz) + soundfile.write(tmp .wav)
        ↓
  VLMSFTTrainer messages format (same as audio_asr.py)
        ↓
  FastVisionModel + LoRA → sinhala_asr_lora/
```

## File Changes Map

| File | Action | Description |
|------|--------|-------------|
| `sinhala_asr.py` | Create | Sinhala ASR fine-tune script |
| `transcribe.py` | Modify | Add `--adapter` and `--prompt` argparse flags |
| `README.md` | Modify | Add Sinhala ASR section |

## Data Model Changes

None — same message format as `audio_asr.py`:
```python
{"messages": [
  {"role": "user", "content": [
    {"type": "audio", "audio": "/tmp/xxx.wav"},
    {"type": "text",  "text": "Transcribe this audio."},
  ]},
  {"role": "assistant", "content": [
    {"type": "text", "text": "<sinhala sentence>"},
  ]},
]}
```

## API Changes

`transcribe.py` CLI change (backwards compatible — old positional arg still works):

```bash
# Before
python transcribe.py /path/to/audio.wav

# After
python transcribe.py /path/to/audio.wav --adapter sinhala_asr_lora --prompt "Transcribe this audio."
```

## Key Decisions

- **Cap at 200 samples** — Common Voice Sinhala train split may be small; 200 gives fast iteration. Easy to change via a `--max-samples` flag or env var later.
- **Temp files cleaned up** — same pattern as `audio_asr.py`; no leftover WAVs.
- **`librosa.resample` not `torchaudio`** — already available, no new deps.
- **HF auth gate** — call `HfApi().whoami()` at script start; if unauthenticated, print `hf auth login` instruction and exit(1).
- **`strict=False` in `from_pretrained`** — required per mlx_vlm 0.6.3 bug (logged in learnings).

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Common Voice Sinhala train split is tiny (<50 clips) | Medium | Medium | Use all available; log actual count |
| HF dataset access requires accepting terms on the website | High | Low | Document clearly in README |
| `librosa.resample` introduces audio artifacts | Low | Low | Acceptable for fine-tuning demo |
