"""
Interactive Sinhala ASR Dataset Recorder

Records audio clips paired with Sinhala sentences for fine-tuning.

Usage:
    uv run python record_dataset.py                        # use built-in sentences
    uv run python record_dataset.py --sentences my.tsv    # use custom sentence list
    uv run python record_dataset.py --list                 # list recorded clips so far

Controls:
    Enter  — start recording
    Enter  — stop recording
    s      — skip this sentence
    q      — quit and save progress
"""

import argparse
import csv
import os
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf

DATA_DIR = "data"
SENTENCES_TSV = os.path.join(DATA_DIR, "sentences.tsv")
SAMPLE_RATE = 16000
CHANNELS = 1

SEED_SENTENCES = [
    "ආයුබෝවන්",
    "ඔබේ නම කුමක්ද?",
    "මගේ නම චන්දිම.",
    "අද හොඳ දවසක්.",
    "කාලගුණය ඉතා සුන්දරයි.",
    "මම ශ්‍රී ලංකාවේ ජීවත් වෙමි.",
    "ඔබ කොහෙද යන්නේ?",
    "කරුණාකර ඔබේ වයස කියන්න.",
    "ස්තූතියි, ඔබට ද සුභ දිනයක්.",
    "මම පාන් ගන්නට කඩේ ගියා.",
    "ශ්‍රී ලංකාව සුන්දර රටක්.",
    "කොළඹ ශ්‍රී ලංකාවේ අගනුවරයි.",
    "මගේ ගෙදර ගාල්ල ළඟ.",
    "අද රාත්‍රිය ඉතා සීතලයි.",
    "හෙට කෝච්චියෙන් යනවා.",
    "ඔබ කොතැනින්ද?",
    "ළමයා ඉස්කෝලේ ගිහිල්ලා.",
    "ඇගේ නම නිලූෆාර්.",
    "අපි එකට ක්‍රිකට් ක්‍රීඩා කළා.",
    "දිය ටිකක් ගෙනෙල්ලා දෙන්න.",
    "රෝහලට යා යුතුයි.",
    "ගෙදරට ගිහිල්ලා ආයිත් එනවා.",
    "කෑම ඉතා රසවත්.",
    "ඔහු ගණිතය ඉගෙන ගනී.",
    "ගහේ ඵල ඉදිලා.",
    "හිරු උදා වෙලා.",
    "වැස්ස ඇදලා.",
    "දරුවා හඬනවා.",
    "නිදා ගන්නට ගියා.",
    "ආහාරය කෑවා.",
    "ගේ හදනවා.",
    "ගමේ ජනයා සතුටින් ජීවත් වෙති.",
    "ඇගෙ ළමා දෙදෙනෙකු ඉන්නවා.",
    "ගමට ගිහිල්ලා ආවා.",
    "ඔවුන් ගෙදර ඉන්නවා.",
    "ශිෂ්‍යයෝ ඉගෙනීමේ යෙදී සිටිති.",
    "ගුරුවරිය ළමයින්ට ඉගැන්වූවා.",
    "ළිඳේ ජලය සීතලයි.",
    "ගොවියා ගොවිතැන් කරනවා.",
    "රාජ්‍යය ජනතාව රකිනවා.",
    "ගුවන් යානය ළඟා වෙනවා.",
    "ඔහු පොතක් කියවනවා.",
    "නාට්‍යයක් බලනවා.",
    "රූපවාහිනිය බලනවා.",
    "ගීතයක් ගැයුවා.",
    "කෙළිලොළු ළමයා නටනවා.",
    "ළිළිත් ගිටාර් වාදනය කරනවා.",
    "ඇය ඇඳුම් මහනවා.",
    "ඔහු කිරිල්ලියෙකු ඇදගෙන ගිහිල්ලා.",
    "ඉදිරි සතියේ විභාගය.",
]


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


def load_sentences(path):
    if not os.path.exists(path):
        return list(SEED_SENTENCES)
    sentences = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                sentences.append(line)
    return sentences


def already_recorded():
    if not os.path.exists(SENTENCES_TSV):
        return set()
    done = set()
    with open(SENTENCES_TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            done.add(row.get("sentence", "").strip())
    return done


def next_clip_index():
    existing = [
        f for f in os.listdir(DATA_DIR) if f.startswith("si_") and f.endswith(".wav")
    ]
    if not existing:
        return 1
    nums = []
    for name in existing:
        try:
            nums.append(int(name[3:7]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 1


def append_tsv(audio_path, sentence):
    write_header = not os.path.exists(SENTENCES_TSV)
    with open(SENTENCES_TSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        if write_header:
            writer.writerow(["audio_path", "sentence"])
        writer.writerow([audio_path, sentence])


def list_recordings():
    if not os.path.exists(SENTENCES_TSV):
        print("No recordings yet.")
        return
    with open(SENTENCES_TSV) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    print(f"\n{len(rows)} recording(s) in {SENTENCES_TSV}:\n")
    for i, row in enumerate(rows, 1):
        print(f"  {i:3d}. {row['audio_path']}  →  {row['sentence'][:60]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Record Sinhala ASR dataset")
    parser.add_argument("--sentences", default=None, help="Path to a plain-text file with one sentence per line")
    parser.add_argument("--list", action="store_true", help="List recorded clips and exit")
    args = parser.parse_args()

    if args.list:
        list_recordings()
        return

    sentence_source = args.sentences if args.sentences else os.path.join(DATA_DIR, "extra_sentences.txt")
    all_sentences = load_sentences(sentence_source) if args.sentences else list(SEED_SENTENCES)
    done = already_recorded()

    pending = [s for s in all_sentences if s.strip() not in done]
    if not pending:
        print("All sentences already recorded! Use --list to review.")
        return

    print("\n=== Sinhala ASR Dataset Recorder ===")
    print(f"Sentences to record: {len(pending)}")
    print("Controls: Enter=start/stop  s=skip  q=quit\n")

    recorder = AudioRecorder()
    idx = next_clip_index()

    for i, sentence in enumerate(pending):
        print(f"[{i+1}/{len(pending)}] {sentence}")

        while True:
            key = input("  Press Enter to record (s=skip, q=quit): ").strip().lower()
            if key == "q":
                print(f"\nSaved {idx - next_clip_index() + (idx - 1)} clips. Run again to continue.")
                return
            if key == "s":
                print("  Skipped.\n")
                break

            print("  🔴 Recording... press Enter to stop.")
            recorder.start()
            input()
            audio = recorder.stop()

            duration = len(audio) / SAMPLE_RATE
            if duration < 0.3:
                print("  Too short (<0.3s). Try again.\n")
                continue

            wav_path = os.path.join(DATA_DIR, f"si_{idx:04d}.wav")
            sf.write(wav_path, audio, SAMPLE_RATE)
            append_tsv(wav_path, sentence)
            print(f"  Saved: {wav_path}  ({duration:.1f}s)\n")
            idx += 1
            break

    print(f"\nDone! Recorded clips are in {DATA_DIR}/")
    print(f"Now fine-tune with:\n  uv run python sinhala_asr.py --local")


if __name__ == "__main__":
    main()
