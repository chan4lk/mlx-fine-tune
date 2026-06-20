"""
Synthesize Sinhala speech from text using a fine-tuned Spark-TTS adapter.

Single string:
    uv run python synthesize.py "ආයුබෝවන්"
    uv run python synthesize.py "ආයුබෝවන්" --out hello.wav

Text file (one line per clip → output_001.wav, output_002.wav, ...):
    uv run python synthesize.py --file sinhala.txt
    uv run python synthesize.py --file sinhala.txt --out audio/clip.wav
"""

import argparse
import os

import soundfile as sf
from mlx_tune import FastTTSModel


def out_path_for(base: str, index: int, total: int) -> str:
    if total == 1:
        return base
    stem, ext = os.path.splitext(base)
    return f"{stem}_{index:03d}{ext}"


def main():
    parser = argparse.ArgumentParser(description="Synthesize Sinhala speech from text")
    parser.add_argument("text", nargs="?", help="Sinhala text to synthesize (inline)")
    parser.add_argument("--file", metavar="PATH",
                        help="Text file to synthesize — each non-empty line becomes a separate WAV")
    parser.add_argument("--adapter", default="sinhala_tts_lora",
                        help="LoRA adapter directory (default: sinhala_tts_lora)")
    parser.add_argument("--out", default="output.wav",
                        help="Output WAV path; for --file with multiple lines, used as stem: output_001.wav, output_002.wav, ... (default: output.wav)")
    args = parser.parse_args()

    if not args.text and not args.file:
        parser.error("provide either a text argument or --file PATH")

    if args.file:
        if not os.path.exists(args.file):
            print(f"ERROR: File '{args.file}' not found.")
            raise SystemExit(1)
        with open(args.file, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            print("ERROR: File is empty or has no non-empty lines.")
            raise SystemExit(1)
    else:
        if not args.text.strip():
            print("ERROR: text argument is empty.")
            raise SystemExit(1)
        lines = [args.text.strip()]

    use_adapter = args.adapter and os.path.exists(args.adapter)
    if args.adapter and not use_adapter:
        print(f"WARNING: Adapter '{args.adapter}' not found — running base model only.")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Loading model with adapter: {args.adapter} ...")
    model, tokenizer = FastTTSModel.from_pretrained(  # noqa: F841
        "mlx-community/Spark-TTS-0.5B-bf16",
    )
    if use_adapter:
        model.load_adapter(args.adapter)
        print(f"Adapter: {args.adapter}")
    else:
        print("Adapter: none (base model)")
    FastTTSModel.for_inference(model)
    print(f"Ready. Synthesizing {len(lines)} clip(s).\n")

    for i, line in enumerate(lines, 1):
        wav_path = out_path_for(args.out, i, len(lines))
        print(f"[{i}/{len(lines)}] {line[:60]}")
        audio = model.generate(text=line, max_tokens=1024)
        sf.write(wav_path, audio, 16000)
        print(f"  Saved: {wav_path}")

    print(f"\nDone. {len(lines)} file(s) written.")


if __name__ == "__main__":
    main()
