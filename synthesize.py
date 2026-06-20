"""
Synthesize Sinhala speech from text using a fine-tuned TTS adapter.

Single string:
    uv run python synthesize.py "ආයුබෝවන්"
    uv run python synthesize.py "ආයුබෝවන්" --model outetts --out-dir hello/

Text file (one line per clip → synthesized/output_001.wav, ...):
    uv run python synthesize.py --file sinhala.txt --model spark
    uv run python synthesize.py --file sinhala.txt --out-dir audio/

TSV file (audio_path + sentence columns → uses audio basename as output name):
    uv run python synthesize.py --tsv recordings/transcriptions-001.tsv
    uv run python synthesize.py --tsv recordings/transcriptions-001.tsv --model outetts --out-dir synthesized/
"""

import argparse
import csv
import os

import soundfile as sf
from mlx_tune import FastTTSModel

MODELS = {
    "spark": {
        "model_id": "mlx-community/Spark-TTS-0.5B-bf16",
        "codec": None,
        "adapter": "sinhala_tts_lora",
    },
    "outetts": {
        "model_id": "mlx-community/Llama-OuteTTS-1.0-1B-8bit",
        "codec": None,
        "adapter": "outetts_lora",
    },
    "qwen3": {
        "model_id": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
        "codec": None,
        "adapter": "qwen3_tts_lora",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Synthesize Sinhala speech from text")
    parser.add_argument("text", nargs="?", help="Sinhala text to synthesize (inline)")
    parser.add_argument("--file", metavar="PATH",
                        help="Text file — each non-empty line becomes a separate WAV")
    parser.add_argument("--tsv", metavar="PATH",
                        help="TSV file with audio_path + sentence columns (same format as speak.py output)")
    parser.add_argument("--model", default="spark", choices=list(MODELS),
                        help="TTS model to use (default: spark)")
    parser.add_argument("--out-dir", default="synthesized",
                        help="Output directory for generated WAV files (default: synthesized/)")
    args = parser.parse_args()

    if not args.text and not args.file and not args.tsv:
        parser.error("provide one of: text argument, --file PATH, or --tsv PATH")

    # Build list of (output_filename, text) pairs
    clips = []

    if args.tsv:
        if not os.path.exists(args.tsv):
            print(f"ERROR: TSV '{args.tsv}' not found.")
            raise SystemExit(1)
        with open(args.tsv, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                audio_path = row.get("audio_path", "").strip()
                sentence = row.get("sentence", "").strip()
                if not sentence:
                    continue
                basename = os.path.basename(audio_path) if audio_path else ""
                if not basename:
                    basename = f"output_{len(clips)+1:03d}.wav"
                clips.append((basename, sentence))
        if not clips:
            print("ERROR: TSV has no usable rows.")
            raise SystemExit(1)

    elif args.file:
        if not os.path.exists(args.file):
            print(f"ERROR: File '{args.file}' not found.")
            raise SystemExit(1)
        with open(args.file, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            print("ERROR: File is empty or has no non-empty lines.")
            raise SystemExit(1)
        for i, line in enumerate(lines, 1):
            clips.append((f"output_{i:03d}.wav", line))

    else:
        text = args.text.strip()
        if not text:
            print("ERROR: text argument is empty.")
            raise SystemExit(1)
        clips.append(("output.wav", text))

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = MODELS[args.model]
    adapter_dir = cfg["adapter"]
    use_adapter = os.path.exists(adapter_dir)
    if not use_adapter:
        print(f"WARNING: Adapter '{adapter_dir}' not found — running base model only.")

    print(f"Loading model: {args.model} ...")
    from_pretrained_kwargs = {"max_seq_length": 2048}
    if cfg["codec"]:
        from_pretrained_kwargs["codec_model"] = cfg["codec"]
    model, tokenizer = FastTTSModel.from_pretrained(cfg["model_id"], **from_pretrained_kwargs)  # noqa: F841
    if use_adapter:
        model.load_adapter(adapter_dir)
        print(f"Adapter: {adapter_dir}")
    else:
        print("Adapter: none (base model)")
    FastTTSModel.for_inference(model)
    print(f"Ready. {len(clips)} clip(s) → {args.out_dir}/\n")

    skipped = 0
    generated = 0
    for i, (filename, text) in enumerate(clips, 1):
        wav_path = os.path.join(args.out_dir, filename)
        if os.path.exists(wav_path):
            print(f"[{i}/{len(clips)}] Skipped (exists): {wav_path}")
            skipped += 1
            continue
        print(f"[{i}/{len(clips)}] {text[:60]}")
        audio = model.generate(text=text, max_tokens=1024)
        sf.write(wav_path, audio, model.sample_rate)
        print(f"  Saved: {wav_path}")
        generated += 1

    print(f"\nDone. {generated} generated, {skipped} skipped.")


if __name__ == "__main__":
    main()
