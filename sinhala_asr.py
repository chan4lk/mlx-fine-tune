"""
Sinhala ASR Fine-Tuning with Gemma 4

Two dataset modes:
  --local  : use recordings from record_dataset.py (data/sentences.tsv)
  (default): use Mozilla Common Voice 17.0 (requires HF login + dataset agreement)

Usage:
    python sinhala_asr.py --local              # your own voice recordings
    python sinhala_asr.py                      # Mozilla Common Voice 17.0
"""

import argparse
import csv
import os
import shutil
import tempfile

import numpy as np
import soundfile as sf
import librosa
from huggingface_hub import HfApi
from datasets import load_dataset

from mlx_tune import FastVisionModel, UnslothVisionDataCollator, VLMSFTTrainer
from mlx_tune.vlm import VLMSFTConfig


MAX_SAMPLES = 200
TARGET_SR = 16000
DEFAULT_LOCAL_TSV = os.path.join("data", "sentences.tsv")


def check_hf_auth():
    try:
        user = HfApi().whoami()
        print(f"HuggingFace: logged in as {user['name']}")
    except Exception:
        print("ERROR: Not logged in to HuggingFace.")
        print("Run:  hf auth login")
        print("Then accept Common Voice terms at:")
        print("  https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0")
        raise SystemExit(1)


def load_local_samples(tsv_path):
    if not os.path.exists(tsv_path):
        print(f"ERROR: {tsv_path} not found.")
        print("Record your own voice first:\n  uv run python record_dataset.py")
        print("Or use speak.py to record + transcribe:\n  uv run python speak.py")
        raise SystemExit(1)
    samples = []
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            path = row.get("audio_path", "").strip()
            sentence = row.get("sentence", "").strip()
            if path and sentence and os.path.exists(path):
                samples.append({"audio_path": path, "sentence": sentence})
    print(f"Loaded {len(samples)} local recordings from {tsv_path}")
    return samples


def load_sinhala_samples(audio_dir: str):
    print("Loading Mozilla Common Voice 17.0 Sinhala dataset...")
    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "si",
        split=f"train[:{MAX_SAMPLES}]",
        trust_remote_code=True,
    )

    samples = []
    skipped = 0
    for i, row in enumerate(ds):
        sentence = (row.get("sentence") or "").strip()  # type: ignore[union-attr]
        if not sentence:
            skipped += 1
            continue

        audio_info = row["audio"]  # type: ignore[index]
        audio_array = np.array(audio_info["array"], dtype=np.float32)
        orig_sr = audio_info["sampling_rate"]

        if orig_sr != TARGET_SR:
            audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=TARGET_SR)

        wav_path = os.path.join(audio_dir, f"si_{i:04d}.wav")
        sf.write(wav_path, audio_array, TARGET_SR)
        samples.append({"audio_path": wav_path, "sentence": sentence})

    print(f"Loaded {len(samples)} samples ({skipped} skipped — no transcription)")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Sinhala ASR fine-tuning with Gemma 4")
    parser.add_argument("--local", action="store_true",
                        help="Use local recordings from data/sentences.tsv")
    parser.add_argument("--tsv", default=None,
                        help="Path to a custom TSV (audio_path + sentence columns); implies --local")
    args = parser.parse_args()
    if args.tsv:
        args.local = True

    tsv_path = args.tsv or DEFAULT_LOCAL_TSV
    print("=" * 70)
    print("GEMMA 4 SINHALA ASR FINE-TUNING")
    if args.local:
        print(f"Dataset: local recordings ({tsv_path})")
    else:
        print("Dataset: Mozilla Common Voice 17.0")
    print("=" * 70)

    if not args.local:
        check_hf_auth()

    print("\n[Step 1] Loading Gemma 4 E4B model...")
    # strict=False: mlx_vlm 0.6.3 drops strict kwarg; Gemma 4 E4B KV-shared
    # layers (24-41) have quantized checkpoint weights the architecture omits.
    model, processor = FastVisionModel.from_pretrained(
        "mlx-community/gemma-4-e4b-it-4bit",
        load_in_4bit=True,
        strict=False,
    )

    print("\n[Step 2] Adding LoRA adapters...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_audio_layers=False,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )

    print("\n[Step 3] Preparing Sinhala ASR dataset...")
    audio_dir = None
    try:
        if args.local:
            samples = load_local_samples(tsv_path)
        else:
            audio_dir = tempfile.mkdtemp(prefix="sinhala_asr_")
            samples = load_sinhala_samples(audio_dir)

        if not samples:
            print("No samples with transcriptions found. Exiting.")
            return

        dataset = [
            {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "audio", "audio": s["audio_path"]},
                        {"type": "text",  "text": "Transcribe this audio."},
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": s["sentence"]},
                    ]},
                ]
            }
            for s in samples
        ]
        print(f"Dataset: {len(dataset)} samples")

        print("\n[Step 4] Training...")
        FastVisionModel.for_training(model)

        trainer = VLMSFTTrainer(
            model=model,
            tokenizer=processor,
            data_collator=UnslothVisionDataCollator(model, processor),
            train_dataset=dataset,
            args=VLMSFTConfig(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=50,
                learning_rate=2e-4,
                logging_steps=5,
                optim="adam",
                weight_decay=0.001,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs_sinhala_asr",
                report_to="none",
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_length=512,
            ),
        )

        trainer_stats = trainer.train()
        print(f"\nTraining metrics: {trainer_stats.metrics}")

        print("\n[Step 5] Saving LoRA adapters...")
        model.save_pretrained("sinhala_asr_lora")
        print("Saved to sinhala_asr_lora/")

    finally:
        if audio_dir:
            shutil.rmtree(audio_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("Done! Gemma 4 Sinhala ASR fine-tuning complete.")
    print("=" * 70)
    print("\nTo transcribe Sinhala audio:")
    print("  uv run python transcribe.py <audio.wav> --adapter sinhala_asr_lora")


if __name__ == "__main__":
    main()
