"""
Sinhala ASR Fine-Tuning with Gemma 4 + Mozilla Common Voice

Fine-tune Gemma 4 E4B on validated Sinhala speech from Mozilla Common Voice 17.0.

Requirements:
  - HuggingFace account with Common Voice dataset access
  - Run `hf auth login` once, then accept terms at:
    https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0

Usage:
    python sinhala_asr.py
"""

import os
import tempfile
import shutil

import numpy as np
import soundfile as sf
import librosa
from huggingface_hub import HfApi
from datasets import load_dataset

from mlx_tune import FastVisionModel, UnslothVisionDataCollator, VLMSFTTrainer
from mlx_tune.vlm import VLMSFTConfig


MAX_SAMPLES = 200
TARGET_SR = 16000


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
        sentence = row.get("sentence", "").strip()
        if not sentence:
            skipped += 1
            continue

        audio_array = np.array(row["audio"]["array"], dtype=np.float32)
        orig_sr = row["audio"]["sampling_rate"]

        if orig_sr != TARGET_SR:
            audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=TARGET_SR)

        wav_path = os.path.join(audio_dir, f"si_{i:04d}.wav")
        sf.write(wav_path, audio_array, TARGET_SR)
        samples.append({"audio_path": wav_path, "sentence": sentence})

    print(f"Loaded {len(samples)} samples ({skipped} skipped — no transcription)")
    return samples


def main():
    print("=" * 70)
    print("GEMMA 4 SINHALA ASR FINE-TUNING")
    print("=" * 70)

    # ========================================================================
    # Step 1: Auth gate
    # ========================================================================
    check_hf_auth()

    # ========================================================================
    # Step 2: Load Gemma 4 model
    # ========================================================================
    print("\n[Step 1] Loading Gemma 4 E4B model...")
    # strict=False: mlx_vlm 0.6.3 drops strict kwarg; Gemma 4 E4B KV-shared
    # layers (24-41) have quantized checkpoint weights the architecture omits.
    model, processor = FastVisionModel.from_pretrained(
        "mlx-community/gemma-4-e4b-it-4bit",
        load_in_4bit=True,
        strict=False,
    )

    # ========================================================================
    # Step 3: Add LoRA adapters
    # ========================================================================
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

    # ========================================================================
    # Step 4: Prepare Sinhala dataset
    # ========================================================================
    print("\n[Step 3] Preparing Sinhala ASR dataset...")
    audio_dir = tempfile.mkdtemp(prefix="sinhala_asr_")
    try:
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

        # ====================================================================
        # Step 5: Train
        # ====================================================================
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

        # ====================================================================
        # Step 6: Save adapters
        # ====================================================================
        print("\n[Step 5] Saving LoRA adapters...")
        model.save_pretrained("sinhala_asr_lora")
        print("Saved to sinhala_asr_lora/")

    finally:
        shutil.rmtree(audio_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("Done! Gemma 4 Sinhala ASR fine-tuning complete.")
    print("=" * 70)
    print("\nTo transcribe Sinhala audio:")
    print("  uv run python transcribe.py <audio.wav> --adapter sinhala_asr_lora")


if __name__ == "__main__":
    main()
