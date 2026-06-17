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

Fine-tune `Gemma 4 E4B` for automatic speech recognition:

```bash
uv pip install soundfile   # one-time dependency
uv run python audio_asr.py
```

This uses Gemma 4's built-in 12-layer Conformer audio tower to transcribe 16 kHz audio. Replace the synthetic demo audio with real datasets (Common Voice, LibriSpeech, FLEURS) for production use.

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
