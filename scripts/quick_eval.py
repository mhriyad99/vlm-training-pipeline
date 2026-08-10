"""
quick_eval.py

Load a trained LoRA adapter, run inference on a held-out image folder, and save
each prediction as a JSON file. Prints the model's output (next to the verified
label, if you have one) so you can eyeball the tricky rules: abbreviation
handling (পিং/মিং/জং), top_row placement, share copy-forward, দাগ/remarks
structure.

Everything that must match training — prompt text, image resolution, message
order — is read from the same config the training run used, so this cannot
silently drift from what the adapter was trained on.

This is deliberately decoupled from the training script — re-run this any
time you retrain, without touching the training loop.

Usage (run from the repo root):
    uv run python scripts/quick_eval.py
    uv run python scripts/quick_eval.py --adapter outputs/adapters/run1/checkpoint-44
    uv run python scripts/quick_eval.py --limit 1        # smoke test one image

Defaults: adapter from config training.output_dir, images from
dataset/holdout_images, predictions written to dataset/holdout_labels.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Running as `python scripts/quick_eval.py` puts scripts/ on sys.path, not the
# repo root, so `src` would not be importable without this.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep the unsloth import first: it patches transformers/peft/trl on import.
from unsloth import FastVisionModel  # noqa: E402

from PIL import Image  # noqa: E402

from src.utils import load_config, load_prompt, resize_image_width  # noqa: E402


def strip_code_fences(text: str) -> tuple[str, bool]:
    """Remove a ```json ... ``` wrapper if the model added one.

    The prompt asks for bare JSON, but a fenced-yet-correct answer is a
    formatting slip, not an extraction failure — counting it as a parse error
    would understate the model. Returns (text, was_fenced).
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped, False
    lines = stripped.splitlines()
    lines = lines[1:]  # drop the opening ``` (and any language tag)
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip(), True


def parse_prediction(raw: str):
    """Parse the model output. Returns (parsed_or_None, note).

    Falls back to raw_decode so that a valid object followed by trailing junk
    is reported as exactly that, rather than as an opaque parse failure.
    """
    text, was_fenced = strip_code_fences(raw)
    note = "wrapped in ``` fences" if was_fenced else ""
    if not text:
        return None, "empty output"
    try:
        return json.loads(text), note
    except json.JSONDecodeError as e:
        try:
            parsed, end = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError:
            return None, f"{note + '; ' if note else ''}invalid JSON: {e}"
        trailing = len(text) - end
        extra = f"valid JSON + {trailing} trailing char(s) discarded"
        return parsed, f"{note}; {extra}" if note else extra


def run_one(model, processor, image_path, prompt_text, max_new_tokens, image_resize):
    image = Image.open(image_path).convert("RGB")
    # Must match the resolution the collator fed during training, or the model
    # sees a document at a scale it never trained on.
    image = resize_image_width(image, image_resize)

    # Content order must match build_conversations() in src/utils.py — text
    # first, then image. Swapping them is a silent train/inference mismatch.
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image"},
        ]}
    ]
    # enable_thinking=False is REQUIRED, not a style choice. The default leaves
    # an open "<think>\n" after the assistant header, so the model reasons in
    # markdown and never reaches the JSON. Training rendered the assistant turn
    # as "<|im_start|>assistant\n<think>\n\n</think>\n\n{json}", and this flag
    # reproduces that prefix exactly.
    input_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(
        images=image,
        text=input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    n_input = int(inputs["input_ids"].shape[-1])
    # do_sample=False is what actually makes this greedy; passing temperature=0
    # alongside it is contradictory and warns.
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    generated = output_ids[0][n_input:]
    return processor.decode(generated, skip_special_tokens=True), n_input, len(generated)


def main():
    parser = argparse.ArgumentParser(
        description="Run a trained adapter over held-out images and save JSON predictions."
    )
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Config used for training — supplies prompt, image_resize, max_seq_length.")
    parser.add_argument("--adapter", default=None,
                        help="Trained adapter/checkpoint dir (default: training.output_dir from config)")
    parser.add_argument("--images", default="dataset/holdout_images",
                        help="Folder of held-out images to test")
    parser.add_argument("--out", default="dataset/holdout_labels",
                        help="Folder to write predicted JSON into (created if missing)")
    parser.add_argument("--labels", default=None,
                        help="Optional folder of verified ground truth (same stem, .json) to print alongside")
    parser.add_argument("--prompt", default=None,
                        help="Override the prompt file (default: prompt_file from config — matches training)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Default: inference.max_new_tokens from config")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N images")
    args = parser.parse_args()

    config = load_config(args.config)
    adapter = args.adapter or config["training"].get("output_dir", "outputs/adapters/run1")
    max_new_tokens = args.max_new_tokens or config.get("inference", {}).get("max_new_tokens", 2048)
    image_resize = config["image_resize"]

    if args.prompt:
        prompt_text = Path(args.prompt).read_text(encoding="utf-8").strip()
    else:
        # Same loader training used, so the prompt is byte-identical.
        prompt_text = load_prompt(config)

    image_dir = Path(args.images)
    if not image_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {image_dir}")
    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not image_paths:
        raise ValueError(f"No .jpg/.jpeg/.png images found in {image_dir}")
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = Path(args.labels) if args.labels else None

    print(f"Loading adapter from {adapter} ...")
    model, processor = FastVisionModel.from_pretrained(
        adapter,  # Unsloth resolves base model + adapter from adapter_config.json
        load_in_4bit=config.get("load_in_4bit", True),
        # Room for the prompt AND the generated answer. Unsloth's default would
        # truncate a ~3.5k-token prompt before generation even begins.
        max_seq_length=config.get("max_seq_length", 2048) + max_new_tokens,
    )
    FastVisionModel.for_inference(model)

    print(f"{len(image_paths)} image(s) | image_resize={image_resize} | "
          f"max_new_tokens={max_new_tokens} | writing JSON to {out_dir}/")

    ok, failed = [], []
    for img_path in image_paths:
        print(f"\n{'=' * 70}\n{img_path.name}\n{'=' * 70}")

        raw_output, n_input, n_generated = run_one(
            model, processor, img_path, prompt_text, max_new_tokens, image_resize
        )
        print(f"({n_input} input tokens -> {n_generated} generated)")
        if n_generated >= max_new_tokens:
            print(f"⚠️  hit the {max_new_tokens}-token generation limit — output is "
                  f"probably cut off mid-JSON. Raise --max-new-tokens.")

        parsed, note = parse_prediction(raw_output)

        if parsed is not None:
            out_path = out_dir / f"{img_path.stem}.json"
            out_path.write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            ok.append(img_path.stem)
            print(f"PREDICTED (parsed OK{'; ' + note if note else ''}) -> {out_path}")
            print(json.dumps(parsed, ensure_ascii=False, indent=2)[:2000])
        else:
            # Keep the raw text — a parse failure is still the run's only record
            # of what the model actually said.
            raw_path = out_dir / f"{img_path.stem}.raw.txt"
            raw_path.write_text(raw_output, encoding="utf-8")
            failed.append((img_path.stem, note))
            print(f"⚠️  JSON PARSE FAILED ({note}) — raw output saved to {raw_path}")
            print(raw_output[:2000])

        if labels_dir:
            label_path = labels_dir / f"{img_path.stem}.json"
            if label_path.exists():
                gt = json.loads(label_path.read_text(encoding="utf-8"))
                print("\nGROUND TRUTH:")
                print(json.dumps(gt, ensure_ascii=False, indent=2)[:2000])
            else:
                print(f"\n(no ground-truth label found at {label_path})")

    total = len(image_paths)
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {len(ok)}/{total} produced parseable JSON -> {out_dir}/")
    if failed:
        print("Failed:")
        for stem, note in failed:
            print(f"  {stem}: {note}")
    print("Note: this checks the model emits well-formed JSON, not that the "
          "extracted values are correct. Compare against verified labels for that.")


if __name__ == "__main__":
    main()
