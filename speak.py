"""
Record audio then transcribe it immediately.

Usage:
    uv run python speak.py
    uv run python speak.py --adapter sinhala_asr_lora
    uv run python speak.py --adapter gemma4_audio_asr_lora
"""

import argparse
import os
import tempfile
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


def main():
    parser = argparse.ArgumentParser(description="Record and transcribe Sinhala speech")
    parser.add_argument("--adapter", default="sinhala_asr_lora",
                        help="LoRA adapter directory (default: sinhala_asr_lora)")
    parser.add_argument("--prompt", default="Transcribe this audio.",
                        help="Transcription prompt")
    args = parser.parse_args()

    if not os.path.exists(args.adapter):
        print(f"ERROR: Adapter '{args.adapter}' not found.")
        print("Fine-tune first:  uv run python sinhala_asr.py --local")
        raise SystemExit(1)

    print(f"Loading model with adapter: {args.adapter} ...")
    model, processor = FastVisionModel.from_pretrained(
        "mlx-community/gemma-4-e4b-it-4bit",
        load_in_4bit=True,
        strict=False,
    )
    model.load_adapter(args.adapter)
    FastVisionModel.for_inference(model)
    print("Ready.\n")

    recorder = AudioRecorder()

    while True:
        print("Press Enter to start recording (q + Enter to quit):")
        key = input().strip().lower()
        if key == "q":
            break

        print("🔴 Recording... press Enter to stop.")
        recorder.start()
        input()
        audio = recorder.stop()

        duration = len(audio) / SAMPLE_RATE
        if duration < 0.3:
            print("Too short — try again.\n")
            continue

        print(f"Recorded {duration:.1f}s. Transcribing...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            sf.write(wav_path, audio, SAMPLE_RATE)
            response = model.generate(
                audio=wav_path,
                prompt=args.prompt,
                max_tokens=256,
                temperature=0.0,
            )
        finally:
            os.unlink(wav_path)

        print(f"\nTranscription: {response}\n")


if __name__ == "__main__":
    main()
