# Spec: orpheus-tts

## Functional Requirements

- FR1: `orpheus_tts.py` loads `mlx-community/orpheus-3b-0.1-ft-bf16` via `FastTTSModel.from_pretrained` with `codec_model="mlx-community/snac_24khz"` (required — SNAC codec must be named explicitly)
- FR2: `orpheus_tts.py` attaches LoRA adapters (r=16, lora_alpha=16) to all attention + MLP projection layers
- FR3: `orpheus_tts.py` accepts `--tsv PATH` (default `data/sentences.tsv`) and loads samples as numpy float32 arrays via soundfile
- FR4: `orpheus_tts.py` accepts `--resume ADAPTER_DIR` and resumes training using `load_weights(strict=False)`
- FR5: `orpheus_tts.py` trains with `TTSSFTConfig(sample_rate=24000, max_steps=60, learning_rate=2e-4, train_on_completions=True)`
- FR6: `orpheus_tts.py` saves adapter to `orpheus_lora/`
- FR7: `synthesize.py` MODELS dict gains an `orpheus` entry with `codec="mlx-community/snac_24khz"` and `adapter="orpheus_lora"`
- FR8: `synthesize.py --model` choices updated to include `orpheus`
- FR9: README documents Orpheus with run command, RAM (~6–8 GB, 16 GB recommended), quality note, and synthesis example
- FR10: `.gitignore` excludes `orpheus_lora/`

## Non-Functional Requirements

- NFR1: `orpheus_tts.py` follows same ~130-line structure as `qwen3_tts.py`
- NFR2: `synthesize.py` change is one dict entry — `codec` field is non-None for Orpheus, triggering the `codec_model` kwarg path that already exists in the loader

## Acceptance Criteria

- AC1: `uv run python orpheus_tts.py --help` exits 0 and shows `--tsv` and `--resume`
- AC2: `uv run python synthesize.py --help` shows `--model {spark,outetts,qwen3,orpheus}`
- AC3: `orpheus_lora/` is listed in `.gitignore`

## Edge Cases

- EC1: `--resume` with missing `adapters.safetensors` → clear error + exit(1)
- EC2: TSV rows with missing audio files → skip silently
- EC3: `codec_model` path in `synthesize.py` already handles non-None codec (established in outetts-tts) — no new code path needed
