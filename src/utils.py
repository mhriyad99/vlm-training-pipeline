"""
Shared utilities for the vlm-training-pipeline.

Nothing document-specific lives here. Anything that needs to change for a
new document type belongs in configs/config.yaml or the prompt file it
points to, not in this module.
"""

import json
from pathlib import Path

import yaml
from PIL import Image


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load the pipeline config. This is the single source of truth for
    model name, paths, and hyperparameters."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at {path}. Copy configs/config.yaml and point "
            f"--config at it, or edit the default in place."
        )
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_prompt(config: dict) -> str:
    """Load the extraction prompt referenced by the config."""
    prompt_path = Path(config["prompt_file"])
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found at {prompt_path}. Create it and point "
            f"prompt_file in your config at it."
        )
    text = prompt_path.read_text(encoding="utf-8").strip()
    if not text or text.startswith("REPLACE THIS FILE"):
        raise ValueError(
            f"{prompt_path} still contains the placeholder text. Replace it "
            f"with your real extraction prompt before building the dataset."
        )
    return text


def find_label_image_pairs(config: dict) -> list[tuple[Path, Path]]:
    """Match verified label JSONs to their corresponding images.

    Only reads from dataset.labels_dir (verified ground truth). Files in a
    labels_draft/ directory, if present, are intentionally never touched by
    this function — that separation is what makes it structurally hard to
    accidentally train on unverified model output.
    """
    ds_cfg = config["dataset"]
    images_dir = Path(ds_cfg["images_dir"])
    labels_dir = Path(ds_cfg["labels_dir"])
    image_ext = ds_cfg["image_ext"]

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    pairs = []
    label_paths = sorted(labels_dir.glob("*.json"))
    if not label_paths:
        raise ValueError(
            f"No label files found in {labels_dir}. Move verified ground "
            f"truth JSON files here before building the dataset."
        )

    for label_path in label_paths:
        img_path = images_dir / (label_path.stem + image_ext)
        if not img_path.exists():
            print(f"WARNING: no image for {label_path.name} at {img_path}, skipping")
            continue
        pairs.append((img_path, label_path))

    if not pairs:
        raise ValueError(
            "No matching image/label pairs found. Check that filenames in "
            "images_dir and labels_dir share the same stem."
        )

    return pairs


def build_conversations(config: dict) -> list[dict]:
    """Build the Unsloth-format conversation list from verified labels.

    Each element is a chat-style conversation: the extraction prompt + image
    as the user turn, and the ground truth JSON (as a string) as the
    assistant turn.
    """
    prompt_text = load_prompt(config)
    pairs = find_label_image_pairs(config)

    conversations = []
    for img_path, label_path in pairs:
        image = Image.open(img_path).convert("RGB")
        with open(label_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)
        ground_truth_str = json.dumps(ground_truth, ensure_ascii=False)

        conversation = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image", "image": image},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ground_truth_str}],
                },
            ]
        }
        conversations.append(conversation)

    print(f"Built {len(conversations)} training example(s) from {len(pairs)} verified label(s)")
    return conversations
