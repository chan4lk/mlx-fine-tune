# Spec: qwen3-tts

## Functional Requirements

- FR1: `qwen3_tts.py` loads `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` via `FastTTSModel.from_pretrained` without an explicit `codec_model` kwarg
- FR2: `qwen3_tts.py` attaches LoRA adapters (r=16, lora_alpha=16) to all attention + MLP projection layers
- FR3: `qwen3_tts.py` accepts `--tsv PATH` (default `data/sentences.tsv`) and loads samples as numpy float32 arrays via soundfile
- FR4: `qwen3_tts.py` accepts `--resume ADAPTER_DIR` and resumes training using `load_weights(strict=False)`
- FR5: `qwen3_tts.py` trains with `TTSSFTConfig(sample_rate=24000, max_steps=60, learning_rate=2e-4, train_on_completions=True)`
- FR6: `qwen3_tts.py` saves adapter to `qwen3_tts_lora/`
- FR7: `synthesize.py` MODELS dict gains a `qwen3` entry pointing to `Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` and `qwen3_tts_lora` adapter dir
- FR8: `synthesize.py --model` choices updated to include `qwen3`
- FR9: README documents Qwen3-TTS with run command, RAM (~5–6 GB), multilingual note, and synthesis example
- FR10: `.gitignore` excludes `qwen3_tts_lora/`

## Non-Functional Requirements

- NFR1: `qwen3_tts.py` follows same ~130-line structure as `outetts_tts.py`
- NFR2: `synthesize.py` change is one dict entry + updating `choices` list — no structural changes
- NFR3: No new dependencies

## Acceptance Criteria

- AC1: `uv run python qwen3_tts.py --help` exits 0 and shows `--tsv` and `--resume`
- AC2: `uv run python synthesize.py --help` shows `--model {spark,outetts,qwen3}`
- AC3: `qwen3_tts_lora/` is listed in `.gitignore`

## Edge Cases

- EC1: `--resume` with missing `adapters.safetensors` → clear error + exit(1)
- EC2: TSV rows with missing audio files → skip silently
