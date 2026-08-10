"""
Run the fine-tuned adapter on a new image.

Usage:
    uv run python src/infer.py --config configs/config.yaml --image path/to/image.jpg --adapter outputs/adapters/run1
"""

import argparse

from PIL import Image

from utils import load_config, load_prompt, resize_image_width


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

    inf_cfg = config.get("inference", {})
    max_new_tokens = inf_cfg.get("max_new_tokens", 2048)

    print(f"Loading base model {config['base_model']} with adapter {adapter_path}")
    model, tokenizer = FastVisionModel.from_pretrained(
        adapter_path,  # Unsloth resolves base model + adapter automatically when saved together
        load_in_4bit=config.get("load_in_4bit", True),
        # Room for prompt AND generation: the prompt alone is ~6700 tokens, so
        # the training cap would be exceeded mid-answer. Cheap to raise here —
        # only 8 of 32 layers are full-attention, so the KV cache stays small.
        max_seq_length=config.get("max_seq_length", 2048) + max_new_tokens,
    )
    FastVisionModel.for_inference(model)

    prompt_text = load_prompt(config)
    image = Image.open(args.image).convert("RGB")
    # Match the resolution the collator fed during training. Skipping this would
    # send a 4096-wide scan as 9216 visual tokens where training saw 2304.
    image = resize_image_width(image, config["image_resize"])

    # Content order must match build_conversations() in utils.py — text first,
    # then image. Swapping them is a silent train/inference mismatch.
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}, {"type": "image"}],
        }
    ]
    # enable_thinking=False is required for parity with training: the default
    # leaves an open "<think>\n" after the assistant header, so the model
    # produces markdown reasoning instead of the JSON it was trained to emit.
    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    # temperature 0 means greedy; do_sample=False is how you actually get that.
    # Leaving do_sample at its default samples from a near-0 temperature instead,
    # which is neither deterministic nor reproducible.
    temperature = inf_cfg.get("temperature", 0.0)
    sampling = {"do_sample": False} if temperature <= 0 else {
        "do_sample": True, "temperature": temperature,
    }
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        **sampling,
    )
    result = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    print("\n--- Model output ---")
    print(result)


if __name__ == "__main__":
    main()
