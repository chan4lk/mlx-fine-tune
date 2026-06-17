from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
from datasets import load_dataset
def main():
    # Load any HuggingFace model (1B model for quick start)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
    )

    # Load a dataset (or create your own)
    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:100]")

    # Train with SFTTrainer (same API as TRL!)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            output_dir="outputs",
            per_device_train_batch_size=2,
            learning_rate=2e-4,
            max_steps=50,
        ),
    )
    trainer.train()

    # Save (same API as Unsloth!)
    model.save_pretrained("lora_model")  # Adapters only
    model.save_pretrained_merged("merged", tokenizer)  # Full model (16-bit)
    model.save_pretrained_gguf("model", tokenizer, dequantize=True)  # GGUF (see note below)

if __name__ == "__main__":
    main()
