import sys
from mlx_tune import FastVisionModel

audio_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sanasuma_mawatha_3.wav"

model, processor = FastVisionModel.from_pretrained(
    "mlx-community/gemma-4-e4b-it-4bit",
    load_in_4bit=True,
    strict=False,
)

model.load_adapter("gemma4_audio_asr_lora")
FastVisionModel.for_inference(model)

print(f"Transcribing: {audio_path}")
response = model.generate(
    audio=audio_path,
    prompt="Transcribe this audio.",
    max_tokens=512,
    temperature=0.0,
)
print("\nTranscription:")
print(response)
