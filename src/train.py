"""
Fine-tune a vision-language model with Unsloth LoRA on your dataset.

Usage:
    uv run python src/train.py --config configs/config.yaml
    uv run python src/train.py --probe 3      # measure peak VRAM first

Everything that changes between projects (base model, LoRA rank, batch size,
output path, etc.) comes from the config file — this script should not need
editing when you switch document types or scale from a small test model to
your real fine-tune.
"""

import argparse
import os
import time

# Must be set before torch initializes CUDA (the unsloth import below is
# deferred, so module import time is early enough). Long sequences on a small
# card fail from allocator fragmentation well before they run out of real VRAM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from utils import build_conversations, find_label_image_pairs, load_config


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a VLM with Unsloth LoRA")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--probe",
        type=int,
        default=None,
        help="Run only N steps on the N HEAVIEST examples and report peak VRAM "
             "and seconds/example, then exit without saving. Use this to check a "
             "long-context config fits before committing hours to a real run.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Imports deferred until after config load / arg parsing so --help is fast
    # and so this file fails with a clear config error before paying Unsloth's
    # import cost.
    import torch
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
    pairs = find_label_image_pairs(config)
    if args.probe:
        # Peak VRAM is set by the LONGEST sequence, so probing the first N
        # examples would measure the wrong thing. Label byte size is a good
        # proxy for response length, which is what varies across examples.
        pairs = sorted(pairs, key=lambda p: p[1].stat().st_size, reverse=True)[: args.probe]
        print(f"PROBE: {args.probe} heaviest example(s): "
              f"{', '.join(label.stem for _, label in pairs)}")
    conversations = build_conversations(config, pairs=pairs)

    FastVisionModel.for_training(model)

    train_cfg = config["training"]
    sft_kwargs = dict(
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
    )
    if args.probe:
        # grad-accum 1 so max_steps == exactly one fwd/bwd per heaviest example.
        # The optimizer step still runs each step, so 8-bit Adam state
        # allocation is included in the measured peak.
        sft_kwargs.update(
            gradient_accumulation_steps=1,
            max_steps=args.probe,
            num_train_epochs=1,
            warmup_steps=1,
            save_strategy="no",
            logging_steps=1,
        )

    trainer = SFTTrainer(
        model=model,
        # TRL >= 0.20 renamed this from `tokenizer`; the unsloth-patched trainer
        # has no `tokenizer` parameter and no **kwargs to absorb it, so passing
        # the old name raises TypeError before training starts.
        processing_class=tokenizer,
        data_collator=UnslothVisionDataCollator(
            model,
            tokenizer,
            # Feed the model a readable image. Without this, unsloth falls back
            # to 512 px (the model config has no default image_size), which
            # turns a 4096x2304 scan into 144 visual tokens.
            resize=config["image_resize"],
            train_on_responses_only=True,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        ),
        train_dataset=conversations,
        args=SFTConfig(**sft_kwargs),
    )

    if args.probe:
        torch.cuda.reset_peak_memory_stats()
    print("Starting training...")
    started = time.time()
    trainer.train()
    elapsed = time.time() - started

    if args.probe:
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n--- PROBE RESULT ({args.probe} heaviest example(s), "
              f"max_seq_length={config.get('max_seq_length')}, "
              f"image_resize={config['image_resize']}) ---")
        print(f"  peak allocated : {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"  peak reserved  : {torch.cuda.max_memory_reserved() / 1e9:.2f} GB "
              f"of {total_vram:.2f} GB total")
        per_example = elapsed / args.probe
        print(f"  sec/example    : {per_example:.1f}")
        n_examples = len(find_label_image_pairs(config))
        epochs = train_cfg.get("num_train_epochs", 3)
        eta_h = per_example * n_examples * epochs / 3600
        print(f"  full-run ETA   : ~{eta_h:.1f} h "
              f"({n_examples} examples x {epochs} epochs)")
        print("\nIt fit. Re-run without --probe for the real thing. If reserved "
              "peak is near total, lower image_resize in the config (2048 -> 1536) "
              "and probe again.")
        return

    output_dir = train_cfg.get("output_dir", "outputs/adapters/run1")
    print(f"Saving adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Done in {elapsed / 3600:.2f} h.")


if __name__ == "__main__":
    main()
