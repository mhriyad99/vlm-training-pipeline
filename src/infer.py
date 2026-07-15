"""
Run the fine-tuned adapter on a new image.

Usage:
    uv run python src/infer.py --config configs/config.yaml --image path/to/image.jpg --adapter outputs/adapters/run1
"""

import argparse

from PIL import Image

from utils import load_config, load_prompt


def main():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned VLM adapter")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--image", required=True, help="Path to the image to run extraction on")
    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to the trained LoRA adapter (defaults to training.output_dir in config)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    adapter_path = args.adapter or config["training"].get("output_dir", "outputs/adapters/run1")

    from unsloth import FastVisionModel

    print(f"Loading base model {config['base_model']} with adapter {adapter_path}")
    model, tokenizer = FastVisionModel.from_pretrained(
        adapter_path,  # Unsloth resolves base model + adapter automatically when saved together
        load_in_4bit=config.get("load_in_4bit", True),
        max_seq_length=config.get("max_seq_length", 2048),
    )
    FastVisionModel.for_inference(model)

    prompt_text = load_prompt(config)
    image = Image.open(args.image).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    inf_cfg = config.get("inference", {})
    output_ids = model.generate(
        **inputs,
        max_new_tokens=inf_cfg.get("max_new_tokens", 2048),
        temperature=max(inf_cfg.get("temperature", 0.0), 1e-4),  # 0.0 not accepted by all samplers
        use_cache=True,
    )
    result = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    print("\n--- Model output ---")
    print(result)


if __name__ == "__main__":
    main()
