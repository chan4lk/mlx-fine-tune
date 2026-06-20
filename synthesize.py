"""
Synthesize Sinhala speech from text using a fine-tuned Spark-TTS adapter.

Usage:
    uv run python synthesize.py "ආයුබෝවන්"
    uv run python synthesize.py "ආයුබෝවන්" --adapter sinhala_tts_lora --out hello.wav
"""

import argparse
import os

import soundfile as sf
from mlx_tune import FastTTSModel


def main():
    parser = argparse.ArgumentParser(description="Synthesize Sinhala speech from text")
    parser.add_argument("text", help="Sinhala text to synthesize")
    parser.add_argument("--adapter", default="sinhala_tts_lora",
                        help="LoRA adapter directory (default: sinhala_tts_lora)")
    parser.add_argument("--out", default="output.wav",
                        help="Output WAV file path (default: output.wav)")
    args = parser.parse_args()

    if not args.text.strip():
        print("ERROR: text argument is empty.")
        raise SystemExit(1)

    if not os.path.exists(args.adapter):
        print(f"ERROR: Adapter '{args.adapter}' not found.")
        print("Fine-tune first:  uv run python sinhala_tts.py")
        raise SystemExit(1)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Loading model with adapter: {args.adapter} ...")
    model, tokenizer = FastTTSModel.from_pretrained(  # noqa: F841
        "mlx-community/Spark-TTS-0.5B-bf16",
    )
    model.load_adapter(args.adapter)
    FastTTSModel.for_inference(model)
    print("Ready.")

    print(f"Synthesizing: {args.text}")
    audio = model.generate(text=args.text, max_tokens=1024)
    sf.write(args.out, audio, 16000)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
