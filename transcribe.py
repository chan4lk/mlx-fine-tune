import argparse
from mlx_tune import FastVisionModel


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with a fine-tuned Gemma 4 ASR model")
    parser.add_argument("audio", help="Path to audio file (WAV, 16kHz recommended)")
    parser.add_argument("--adapter", default="gemma4_audio_asr_lora",
                        help="Path to LoRA adapter directory (default: gemma4_audio_asr_lora)")
    parser.add_argument("--prompt", default="Transcribe this audio.",
                        help="Transcription prompt (default: 'Transcribe this audio.')")
    args = parser.parse_args()

    model, processor = FastVisionModel.from_pretrained(
        "mlx-community/gemma-4-e4b-it-4bit",
        load_in_4bit=True,
        strict=False,
    )

    model.load_adapter(args.adapter)
    FastVisionModel.for_inference(model)

    print(f"Transcribing: {args.audio}")
    print(f"Adapter: {args.adapter}")
    response = model.generate(
        audio=args.audio,
        prompt=args.prompt,
        max_tokens=512,
        temperature=0.0,
    )
    print("\nTranscription:")
    print(response)


if __name__ == "__main__":
    main()
