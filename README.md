# mlx-fine-tune

Fine-tune large language models locally on Apple Silicon using [mlx-tune](https://github.com/unslothai/mlx-tune) with LoRA adapters. Powered by [MLX](https://github.com/ml-explore/mlx) — no GPU required.

## Features

- **LoRA fine-tuning** — efficient adapter-based training with configurable rank and target modules
- **Text generation** — fine-tune Llama 3.2 on instruction datasets (Alpaca)
- **Audio ASR** — fine-tune Gemma 4 for speech-to-text using its built-in audio tower
- **Multiple export formats** — save as LoRA adapters, merged 16-bit model, or GGUF
- **Unsloth-compatible API** — familiar `FastLanguageModel` / `SFTTrainer` interface

## Requirements

- Apple Silicon Mac (M1 or later)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
git clone https://github.com/chan4lk/mlx-fine-tune.git
cd mlx-fine-tune
uv sync
```

## Examples

### Text fine-tuning (Llama 3.2 + Alpaca)

Fine-tune `Llama-3.2-1B-Instruct` on the Alpaca instruction dataset:

```bash
uv run python main.py
```

This will:
1. Load `mlx-community/Llama-3.2-1B-Instruct-4bit` (4-bit quantized)
2. Attach LoRA adapters (`r=16`) to attention projections
3. Train on 100 samples from `yahma/alpaca-cleaned` for 50 steps
4. Save outputs to `lora_model/`, `merged/`, and `model/`

### Audio ASR fine-tuning (Gemma 4)

Fine-tune `Gemma 4 E4B` for automatic speech recognition using its built-in 12-layer Conformer audio tower:

```bash
uv run python audio_asr.py
```

**Requirements:**
- Apple Silicon Mac (M1 or later) — MLX does not support Intel Macs
- ~2 GB free disk space for model download (`mlx-community/gemma-4-e4b-it-4bit`)
- 8 GB RAM minimum (16 GB recommended)

The model is downloaded automatically from HuggingFace on first run. Subsequent runs use the cached copy.

**Model variants:**

| Model | Size | Use case |
|-------|------|----------|
| `gemma-4-e4b-it-4bit` (default) | ~2 GB | Better quality |
| `gemma-4-e2b-it-4bit` | ~1 GB | Lower memory / edge |

To use the smaller E2B variant, edit `audio_asr.py` and replace `gemma-4-e4b-it-4bit` with `gemma-4-e2b-it-4bit`.

**Using real audio datasets** (for production):

The script ships with synthetic audio for a self-contained demo. To fine-tune on real speech data, replace the `generate_synthetic_audio` call with a real dataset loader:

```python
# Common Voice (Mozilla)
from datasets import load_dataset
ds = load_dataset("mozilla-foundation/common_voice_17_0", "en", split="train")

# LibriSpeech
ds = load_dataset("openslr/librispeech_asr", "clean", split="train.100")

# FLEURS (multilingual)
ds = load_dataset("google/fleurs", "en_us", split="train")
```

### Sinhala ASR fine-tuning (Gemma 4 + Mozilla Common Voice)

Fine-tune `Gemma 4 E4B` on validated Sinhala speech from [Mozilla Common Voice 17.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0):

**Prerequisites:**

1. Create a HuggingFace account and accept the Common Voice dataset terms:
   https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0

2. Log in to HuggingFace (run once in the terminal):
   ```bash
   ! hf auth login
   ```

**Run fine-tuning:**

```bash
uv run python sinhala_asr.py
```

This loads up to 200 validated Sinhala clips, resamples audio to 16kHz, and trains for 50 steps (~2 minutes on Apple Silicon M-series). Adapters are saved to `sinhala_asr_lora/`.

**Transcribe with the Sinhala adapter:**

```bash
uv run python transcribe.py /path/to/audio.wav --adapter sinhala_asr_lora
```

The `transcribe.py` script supports any adapter directory via `--adapter`:

```bash
# Sinhala adapter
uv run python transcribe.py audio.wav --adapter sinhala_asr_lora

# Default English adapter (from audio_asr.py)
uv run python transcribe.py audio.wav --adapter gemma4_audio_asr_lora

# Custom prompt
uv run python transcribe.py audio.wav --adapter sinhala_asr_lora --prompt "Transcribe this Sinhala speech."
```

### Sinhala TTS fine-tuning (Spark-TTS)

Fine-tune `Spark-TTS-0.5B` on your own Sinhala voice recordings to synthesize speech from text:

```bash
uv run python sinhala_tts.py
```

Uses `data/sentences.tsv` by default (50 clips, ~3 minutes training on Apple Silicon). Adapters saved to `sinhala_tts_lora/`.

> **Note:** 50 samples produces demo-quality synthesis. Record more clips with `record_dataset.py` for better results.

**Use a custom dataset:**

```bash
uv run python sinhala_tts.py --tsv recordings/transcriptions-001.tsv
```

**Continue training from existing adapter:**

```bash
uv run python sinhala_tts.py --resume sinhala_tts_lora
```

**Synthesize Sinhala speech:**

```bash
uv run python synthesize.py "ආයුබෝවන්"
uv run python synthesize.py "ආයුබෝවන්" --model spark --out-dir synthesized/
```

### Sinhala TTS fine-tuning (OuteTTS 1B, 24 kHz)

Fine-tune `OuteTTS-1.0-1B` (Llama-based, DAC codec) on your Sinhala voice recordings:

```bash
uv run python outetts_tts.py
```

Uses `data/sentences.tsv` by default. Adapters saved to `outetts_lora/`. Sample rate: **24 kHz**.

**RAM requirement:** ~3–4 GB (8-bit quantised model)

**Use a custom dataset:**

```bash
uv run python outetts_tts.py --tsv recordings/transcriptions-001.tsv
```

**Continue training from existing adapter:**

```bash
uv run python outetts_tts.py --resume outetts_lora
```

**Synthesize with OuteTTS:**

```bash
uv run python synthesize.py "ආයුබෝවන්" --model outetts
uv run python synthesize.py --tsv recordings/transcriptions-001.tsv --model outetts --out-dir synthesized/
```

### Sinhala TTS fine-tuning (Qwen3-TTS 1.7B, 24 kHz, multilingual)

Fine-tune `Qwen3-TTS-12Hz-1.7B-VoiceDesign` (Qwen3-based, built-in speech tokenizer) on your Sinhala voice recordings. Natively supports ZH, EN, JA, KO and more — well-suited for low-resource language adaptation.

```bash
uv run python qwen3_tts.py
```

Uses `data/sentences.tsv` by default. Adapters saved to `qwen3_tts_lora/`. Sample rate: **24 kHz**.

**RAM requirement:** ~5–6 GB (bf16 model)

**Use a custom dataset:**

```bash
uv run python qwen3_tts.py --tsv recordings/transcriptions-001.tsv
```

**Continue training from existing adapter:**

```bash
uv run python qwen3_tts.py --resume qwen3_tts_lora
```

**Synthesize with Qwen3-TTS:**

```bash
uv run python synthesize.py "ආයුබෝවන්" --model qwen3
uv run python synthesize.py --tsv recordings/transcriptions-001.tsv --model qwen3 --out-dir synthesized/
```

## Project Structure

```
mlx-fine-tune/
├── main.py             # Llama 3.2 text fine-tuning example
├── audio_asr.py        # Gemma 4 audio ASR fine-tuning (synthetic data)
├── sinhala_asr.py      # Gemma 4 Sinhala ASR fine-tuning (Common Voice)
├── transcribe.py       # Inference: transcribe audio with a LoRA adapter
├── pyproject.toml      # Project dependencies (mlx-tune[audio])
├── sinhala_asr_lora/   # Saved Sinhala LoRA adapters (generated)
├── gemma4_audio_asr_lora/ # Saved English LoRA adapters (generated)
├── lora_model/         # Saved LoRA adapters from text fine-tuning (generated)
├── merged/             # Merged 16-bit model (generated)
├── model/              # GGUF export (generated)
└── outputs/            # Training checkpoints (generated)
```

## Configuration

Key LoRA parameters in `main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r` | 16 | LoRA rank — higher = more capacity |
| `lora_alpha` | 16 | Scaling factor |
| `target_modules` | q/k/v/o_proj | Attention layers to adapt |
| `max_seq_length` | 2048 | Maximum token length |
| `learning_rate` | 2e-4 | Training learning rate |
| `max_steps` | 50 | Number of training steps |

## Dependencies

- [`mlx-tune[audio]`](https://pypi.org/project/mlx-tune/) >= 0.5.1 — MLX-native fine-tuning library

## License

MIT
