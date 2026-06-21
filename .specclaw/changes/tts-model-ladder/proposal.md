# Proposal: tts-model-ladder

**Status:** proposed
**Created:** 2026-06-20

---

## Problem Statement

The project currently fine-tunes one TTS model (Spark-TTS 0.5B). Three additional TTS architectures are available in mlx-tune — OuteTTS 1B, Qwen3-TTS 1.7B, and Orpheus 3B — each with different codecs, strengths, and memory profiles. There is no way to experiment with or compare them on the same Sinhala voice dataset.

---

## Proposed Solution

Add fine-tuning scripts and synthesis support for the three remaining TTS models, ordered by size so each builds on the previous pattern:

| Step | Model | Size | HF Model ID | Codec | Sample Rate |
|------|-------|------|-------------|-------|-------------|
| ✅ done | Spark-TTS | 0.5B | `mlx-community/Spark-TTS-0.5B-bf16` | BiCodec (built-in) | 16 kHz |
| T1 | OuteTTS | 1B | `mlx-community/Llama-OuteTTS-1.0-1B-8bit` | DAC (auto-detected) | 24 kHz |
| T2 | Qwen3-TTS | 1.7B | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` | built-in speech tokenizer | 24 kHz |
| T3 | Orpheus | 3B | `mlx-community/orpheus-3b-0.1-ft-bf16` | SNAC 24 kHz (explicit) | 24 kHz |

### Per-model deliverables (each step)

1. **Fine-tuning script** — `outetts_tts.py` / `qwen3_tts.py` / `orpheus_tts.py`
   - Mirrors `sinhala_tts.py`: `--tsv` (default `data/sentences.tsv`), `--resume`, LoRA r=16
   - Loads WAV arrays via soundfile (avoids TTSDataCollator string bug)
   - Saves adapter to `outetts_lora/` / `qwen3_tts_lora/` / `orpheus_lora/`
   - Only difference per model: `model_name`, optional `codec_model`, `sample_rate` in `TTSSFTConfig`

2. **synthesize.py `--model` flag** — `spark` (default) | `outetts` | `qwen3` | `orpheus`
   - Selects the correct `model_name` + `codec_model` + adapter directory
   - Uses `model.sample_rate` (not hardcoded 16000) when writing WAV

3. **README section** — one entry per model in the Sinhala TTS section

### Key technical differences

| Model | Extra `from_pretrained` kwarg | Notes |
|-------|-------------------------------|-------|
| OuteTTS | none (DAC auto-detected from profile) | 8-bit quantized; Llama arch |
| Qwen3-TTS | none (built-in tokenizer) | multilingual ZH/EN/JA/KO/+; dual embedding path |
| Orpheus | `codec_model="mlx-community/snac_24khz"` | SNAC codec must be named explicitly |

All three share the same `TTSSFTTrainer` / `TTSDataCollator` / `TTSSFTConfig` API as Spark-TTS.

---

## Scope

### In

- `outetts_tts.py`, `qwen3_tts.py`, `orpheus_tts.py` fine-tuning scripts
- `synthesize.py` updated with `--model` flag + `model.sample_rate` for WAV writes
- README entries for each model
- `.gitignore` entries for `outetts_lora/`, `qwen3_tts_lora/`, `orpheus_lora/` adapter weights

### Out

- Changing `record_dataset.py` or `speak.py` (dataset recording is model-agnostic)
- Voice cloning / reference audio features (future)
- Merging adapters or exporting to GGUF
- Training on any dataset other than the existing `data/sentences.tsv` / `recordings/` TSVs

---

## Impact

| Area | Files | Complexity | Risk |
|------|-------|------------|------|
| New scripts | `outetts_tts.py`, `qwen3_tts.py`, `orpheus_tts.py` | low (copy-adapt pattern) | low |
| Modify | `synthesize.py` | low (add `--model` dispatch) | low |
| Modify | `README.md` | low | none |
| Modify | `.gitignore` | trivial | none |

Each script is ~130 lines, nearly identical to `sinhala_tts.py`. The only per-model variation is the model ID, optional codec, and adapter output directory.

---

## Open Questions

1. Should each model get its own inference script (e.g. `synthesize_outetts.py`) instead of a `--model` flag in the shared `synthesize.py`? A single script is simpler but grows with each model.
2. Orpheus 3B requires ~6–8 GB RAM at bf16 — should we document a minimum RAM requirement per model?
3. Should the three changes be separate specclaw changes (one per model) or one change with three waves? One change with three waves keeps the ladder together; separate changes are easier to skip if a model doesn't work.
