"""
Orpheus TTS Fine-Tuning (3B, 24 kHz, SNAC codec)

Fine-tune Orpheus-3B (Llama-based, SNAC 24 kHz codec) on Sinhala voice recordings using LoRA.
Largest model in the TTS ladder — highest quality ceiling on Apple Silicon with 16 GB+ RAM.

Usage:
    python orpheus_tts.py                           # use data/sentences.tsv
    python orpheus_tts.py --tsv <path>              # use a custom TSV
    python orpheus_tts.py --resume orpheus_lora     # continue from existing adapter
"""

import argparse
import csv
import os

import soundfile as sf
from mlx_tune import FastTTSModel, TTSSFTTrainer, TTSSFTConfig, TTSDataCollator

DEFAULT_TSV = os.path.join("data", "sentences.tsv")


def load_samples(tsv_path):
    if not os.path.exists(tsv_path):
        print(f"ERROR: {tsv_path} not found.")
        print("Record your own voice first:\n  uv run python record_dataset.py")
        raise SystemExit(1)
    samples = []
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            path = row.get("audio_path", "").strip()
            sentence = row.get("sentence", "").strip()
            if not (path and sentence and os.path.exists(path)):
                continue
            audio_array, _ = sf.read(path, dtype="float32")
            samples.append({"audio": audio_array, "sentence": sentence})
    print(f"Loaded {len(samples)} recordings from {tsv_path}")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Orpheus 3B TTS fine-tuning")
    parser.add_argument("--tsv", default=DEFAULT_TSV,
                        help=f"TSV with audio_path + sentence columns (default: {DEFAULT_TSV})")
    parser.add_argument("--resume", default=None, metavar="ADAPTER_DIR",
                        help="Continue training from an existing LoRA adapter (e.g. orpheus_lora)")
    args = parser.parse_args()

    print("=" * 70)
    print("ORPHEUS 3B SINHALA TTS FINE-TUNING")
    print(f"Dataset: {args.tsv}")
    if args.resume:
        print(f"Resume:  continuing from {args.resume}")
    print("=" * 70)

    print("\n[Step 1] Loading Orpheus model + SNAC codec...")
    model, tokenizer = FastTTSModel.from_pretrained(
        "mlx-community/orpheus-3b-0.1-ft-bf16",
        max_seq_length=2048,
        codec_model="mlx-community/snac_24khz",
    )

    print("\n[Step 2] Adding LoRA adapters...")
    model = FastTTSModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    if args.resume:
        adapter_weights = os.path.join(args.resume, "adapters.safetensors")
        if not os.path.exists(adapter_weights):
            print(f"ERROR: Adapter weights not found at {adapter_weights}")
            raise SystemExit(1)
        print(f"\n[Step 2b] Resuming from {adapter_weights} ...")
        import mlx.core as mx
        saved = list(mx.load(adapter_weights).items())
        model.load_weights(saved, strict=False)
        print(f"Loaded {len(saved)} adapter weight tensors.")

    print("\n[Step 3] Preparing dataset...")
    samples = load_samples(args.tsv)
    if not samples:
        print("No samples found. Exiting.")
        return
    print(f"Dataset: {len(samples)} samples")

    print("\n[Step 4] Training...")
    FastTTSModel.for_training(model)

    collator = TTSDataCollator(
        model=model,
        tokenizer=tokenizer,
        text_column="sentence",
        audio_column="audio",
    )

    trainer = TTSSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=samples,
        args=TTSSFTConfig(
            output_dir="outputs_orpheus_tts",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            logging_steps=5,
            weight_decay=0.01,
            sample_rate=24000,
            report_to="none",
            train_on_completions=True,
        ),
    )

    trainer_stats = trainer.train()
    print(f"\nTraining metrics: {trainer_stats.metrics}")

    print("\n[Step 5] Saving LoRA adapters...")
    model.save_pretrained("orpheus_lora")
    print("Saved to orpheus_lora/")

    print("\n" + "=" * 70)
    print("Done! Orpheus 3B Sinhala TTS fine-tuning complete.")
    print("=" * 70)
    print("\nTo synthesize Sinhala speech:")
    print("  uv run python synthesize.py 'ආයුබෝවන්' --model orpheus")


if __name__ == "__main__":
    main()
