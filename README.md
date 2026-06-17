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

## Project Structure

```
mlx-fine-tune/
├── main.py           # Llama 3.2 text fine-tuning example
├── audio_asr.py      # Gemma 4 audio ASR fine-tuning example
├── pyproject.toml    # Project dependencies (mlx-tune[audio])
├── lora_model/       # Saved LoRA adapters (generated)
├── merged/           # Merged 16-bit model (generated)
├── model/            # GGUF export (generated)
└── outputs/          # Training checkpoints (generated)
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
