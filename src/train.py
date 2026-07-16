"""
Fine-tune a vision-language model with Unsloth LoRA on your dataset.

Usage:
    uv run python src/train.py --config configs/config.yaml

Everything that changes between projects (base model, LoRA rank, batch size,
output path, etc.) comes from the config file — this script should not need
editing when you switch document types or scale from a small test model to
your real fine-tune.
"""

import argparse

from utils import build_conversations, load_config


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a VLM with Unsloth LoRA")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # Imports deferred until after config load / arg parsing so --help is fast
    # and so this file fails with a clear config error before paying Unsloth's
    # import cost.
    from unsloth import FastVisionModel, is_bf16_supported
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    print(f"Loading base model: {config['base_model']}")
    model, tokenizer = FastVisionModel.from_pretrained(
        config["base_model"],
        load_in_4bit=config.get("load_in_4bit", True),
        max_seq_length=config.get("max_seq_length", 2048),
    )

    lora_cfg = config["lora"]
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=lora_cfg.get("finetune_vision_layers", True),
        finetune_language_layers=lora_cfg.get("finetune_language_layers", True),
        finetune_attention_modules=lora_cfg.get("finetune_attention_modules", True),
        finetune_mlp_modules=lora_cfg.get("finetune_mlp_modules", True),
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias="none",
        random_state=config["training"].get("seed", 3407),
        use_rslora=False,
        loftq_config=None,
    )

    print("Building dataset from verified labels...")
    conversations = build_conversations(config)

    FastVisionModel.for_training(model)

    train_cfg = config["training"]
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(
                      model,
                            tokenizer,
                            train_on_responses_only=True,
                            instruction_part="<|im_start|>user\n",
                            response_part="<|im_start|>assistant\n",
                        ),
        train_dataset=conversations,
        args=SFTConfig(
            per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
            warmup_steps=train_cfg.get("warmup_steps", 5),
            num_train_epochs=train_cfg.get("num_train_epochs", 3),
            learning_rate=train_cfg.get("learning_rate", 2e-4),
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=train_cfg.get("logging_steps", 1),
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=train_cfg.get("seed", 3407),
            output_dir=train_cfg.get("output_dir", "outputs/adapters/run1"),
            save_strategy=train_cfg.get("save_strategy", "epoch"),
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=config.get("max_seq_length", 2048),
        ),
    )

    print("Starting training...")
    trainer.train()

    output_dir = train_cfg.get("output_dir", "outputs/adapters/run1")
    print(f"Saving adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
