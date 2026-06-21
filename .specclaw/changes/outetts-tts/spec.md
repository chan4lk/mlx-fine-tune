# Spec: outetts-tts

## Functional Requirements

- FR1: `outetts_tts.py` loads `mlx-community/Llama-OuteTTS-1.0-1B-8bit` via `FastTTSModel.from_pretrained` without an explicit `codec_model` kwarg
- FR2: `outetts_tts.py` attaches LoRA adapters (r=16, lora_alpha=16) to all attention + MLP projection layers
- FR3: `outetts_tts.py` accepts `--tsv PATH` (default `data/sentences.tsv`) and loads samples as numpy float32 arrays via soundfile
- FR4: `outetts_tts.py` accepts `--resume ADAPTER_DIR` and resumes training using `load_weights(strict=False)` (not `load_adapter`)
- FR5: `outetts_tts.py` trains with `TTSSFTConfig(sample_rate=24000, max_steps=60, learning_rate=2e-4, train_on_completions=True)`
- FR6: `outetts_tts.py` saves adapter to `outetts_lora/`
- FR7: `synthesize.py` gains a `--model` flag with choices `spark` (default) and `outetts`
- FR8: `synthesize.py` dispatches model loading via a dict keyed by model name, each entry holding `model_id`, optional `codec_model`, and `adapter_dir`
- FR9: `synthesize.py` uses `model.sample_rate` (not hardcoded `16000`) when writing WAV files
- FR10: `spark` default in `synthesize.py` produces identical behaviour to before this change
- FR11: README documents OuteTTS with run command, RAM (~3–4 GB), and synthesis example
- FR12: `.gitignore` excludes `outetts_lora/`

## Non-Functional Requirements

- NFR1: `outetts_tts.py` is ~130 lines, same structure as `sinhala_tts.py`
- NFR2: `synthesize.py` dispatch table is a plain dict at module level — no classes
- NFR3: No new dependencies; uses existing `mlx-tune[audio]`, `soundfile`, `mlx.core`

## Acceptance Criteria

- AC1: `uv run python outetts_tts.py --help` exits 0 and shows `--tsv` and `--resume`
- AC2: `uv run python synthesize.py --help` shows `--model {spark,outetts}`
- AC3: `uv run python synthesize.py "test" --model spark` behaves the same as before (loads Spark-TTS, uses `model.sample_rate`)
- AC4: `uv run python synthesize.py "test" --model outetts` loads `Llama-OuteTTS-1.0-1B-8bit`
- AC5: `outetts_lora/` is listed in `.gitignore`

## Edge Cases

- EC1: `--resume` points to a directory without `adapters.safetensors` → print clear error and exit(1)
- EC2: TSV has rows with missing audio files → skip silently, count and report at end
- EC3: `--model spark` with no adapter found → falls back to base model with WARNING (existing behaviour preserved)
- EC4: `--model outetts` with no adapter found → same fallback pattern
