"""
Build the training dataset from verified labels + images.

Usage:
    uv run python src/build_dataset.py --config configs/config.yaml

This does not know anything about khatians, Bangla, or any specific
document type — that all lives in the prompt file and your label JSONs.
Swap the config/prompt/dataset and this script works unchanged.
"""

import argparse

from utils import build_conversations, load_config


def main():
    parser = argparse.ArgumentParser(description="Build training dataset from verified labels")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    conversations = build_conversations(config)

    print(f"\nSample conversation (first example, image omitted):")
    sample = conversations[0]
    for message in sample["messages"]:
        role = message["role"]
        text_parts = [c["text"] for c in message["content"] if c["type"] == "text"]
        for text in text_parts:
            preview = text if len(text) < 200 else text[:200] + "..."
            print(f"  [{role}] {preview}")

    return conversations


if __name__ == "__main__":
    main()
