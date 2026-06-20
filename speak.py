"""
Record audio, transcribe immediately, and save both to a dataset for retraining.

Each recording is saved as a WAV in --out-dir. A TSV (transcriptions.tsv) is
updated with the audio path and model transcription after every clip. Open the
TSV in any spreadsheet/editor to correct wrong transcriptions, then retrain:

    uv run python sinhala_asr.py --tsv recordings/transcriptions.tsv

Usage:
    uv run python speak.py
    uv run python speak.py --adapter sinhala_asr_lora --out-dir recordings
"""

import argparse
import csv
import os
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf

from mlx_tune import FastVisionModel

SAMPLE_RATE = 16000
CHANNELS = 1


class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.frames = []
        self._lock = threading.Lock()

    def _callback(self, indata, _frames, _time, _status):
        if self.recording:
            with self._lock:
                self.frames.append(indata.copy())

    def start(self):
        self.frames = []
        self.recording = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        self.recording = False
        self._stream.stop()
        self._stream.close()
        with self._lock:
            if self.frames:
                return np.concatenate(self.frames, axis=0).flatten()
            return np.array([], dtype=np.float32)


def next_index(out_dir):
    existing = [f for f in os.listdir(out_dir) if f.startswith("rec_") and f.endswith(".wav")]
    if not existing:
        return 1
    nums = []
    for name in existing:
        try:
            nums.append(int(name[4:8]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 1


def append_tsv(tsv_path, audio_path, transcription):
    write_header = not os.path.exists(tsv_path)
    with open(tsv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        if write_header:
            writer.writerow(["audio_path", "sentence"])
        writer.writerow([audio_path, transcription])


def main():
    parser = argparse.ArgumentParser(description="Record, transcribe, and save for retraining")
    parser.add_argument("--adapter", default="sinhala_asr_lora",
                        help="LoRA adapter directory (default: sinhala_asr_lora)")
    parser.add_argument("--out-dir", default="recordings",
                        help="Directory to save WAV files and transcriptions.tsv (default: recordings)")
    parser.add_argument("--prompt", default="Transcribe this audio.",
                        help="Transcription prompt")
    args = parser.parse_args()

    if not os.path.exists(args.adapter):
        print(f"ERROR: Adapter '{args.adapter}' not found.")
        print("Fine-tune first:  uv run python sinhala_asr.py --local")
        raise SystemExit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    tsv_path = os.path.join(args.out_dir, "transcriptions.tsv")

    print(f"Loading model with adapter: {args.adapter} ...")
    model, processor = FastVisionModel.from_pretrained(  # noqa: F841
        "mlx-community/gemma-4-e4b-it-4bit",
        load_in_4bit=True,
        strict=False,
    )
    model.load_adapter(args.adapter)
    FastVisionModel.for_inference(model)
    print(f"Ready. Saving audio + transcriptions to: {args.out_dir}/\n")

    recorder = AudioRecorder()
    idx = next_index(args.out_dir)

    while True:
        print("Press Enter to start recording (q + Enter to quit):")
        key = input().strip().lower()
        if key == "q":
            break

        print("Recording... press Enter to stop.")
        recorder.start()
        input()
        audio = recorder.stop()

        duration = len(audio) / SAMPLE_RATE
        if duration < 0.3:
            print("Too short — try again.\n")
            continue

        wav_path = os.path.join(args.out_dir, f"rec_{idx:04d}.wav")
        sf.write(wav_path, audio, SAMPLE_RATE)
        print(f"Saved: {wav_path} ({duration:.1f}s). Transcribing...")

        transcription = model.generate(
            audio=wav_path,
            prompt=args.prompt,
            max_tokens=256,
            temperature=0.0,
        )

        append_tsv(tsv_path, wav_path, transcription)
        print(f"Transcription: {transcription}")
        print(f"Logged to: {tsv_path}\n")
        idx += 1

    print(f"\nDone. {idx - next_index(args.out_dir) + 1} clips saved.")
    print(f"Review and correct transcriptions in: {tsv_path}")
    print(f"Then retrain with:")
    print(f"  uv run python sinhala_asr.py --tsv {tsv_path}")


if __name__ == "__main__":
    main()
