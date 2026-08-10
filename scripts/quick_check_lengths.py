
"""
quick_check_lengths.py

Answers the real question before picking a training max_seq_length: how many
tokens does each (image + prompt + response) example actually take, and how
many of those are RESPONSE tokens (the only ones that get gradient signal
under train_on_responses_only=True)?

Run this BEFORE launching a training job. It will tell you:
  - the max/mean/percentile total sequence length across your whole dataset
  - the max/mean/percentile RESPONSE-only length
  - how much of each example is image tokens (the knob you fix with resizing,
    not with max_seq_length)
  - whether your config's max_seq_length would truncate any example's response
  - a recommended max_seq_length with headroom

Measurement note: the collator truncates at its own max_seq_length (falling
back to model.max_seq_length when you don't pass one), so measuring through a
collator configured like training's would silently cap every number here at
the very limit we're trying to evaluate. This script therefore collates with
an effectively infinite limit to get TRUE lengths, then compares them against
your target limit in plain arithmetic.

Usage (run from the repo root — the config's paths are relative to cwd):
    uv run python scripts/quick_check_lengths.py
    uv run python scripts/quick_check_lengths.py --max-length 4096
    uv run python scripts/quick_check_lengths.py --limit 10     # fast smoke test
"""

import argparse
import json
import statistics as stats
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Running as `python scripts/quick_check_lengths.py` puts scripts/ on sys.path,
# not the repo root, so `src` would not be importable without this.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep the unsloth import first: it patches transformers/peft/trl on import.
from unsloth import FastVisionModel  # noqa: E402
from unsloth.trainer import UnslothVisionDataCollator  # noqa: E402

from src.utils import build_conversations, find_label_image_pairs, load_config  # noqa: E402

# Large enough that the processor never truncates, so we observe true lengths.
# Must be an int — the collator only enables truncation for int limits.
NO_TRUNCATION = 10_000_000


def percentile(data, p):
    data = sorted(data)
    k = (len(data) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(data) - 1)
    if f == c:
        return data[f]
    return data[f] + (data[c] - data[f]) * (k - f)


def summarize(name, values):
    print(f"\n--- {name} ---")
    print(f"  n       = {len(values)}")
    print(f"  min     = {min(values)}")
    print(f"  mean    = {stats.mean(values):.1f}")
    print(f"  median  = {stats.median(values)}")
    print(f"  p90     = {percentile(values, 90):.0f}")
    print(f"  p99     = {percentile(values, 99):.0f}")
    print(f"  max     = {max(values)}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure true token lengths of the training set before training."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to your existing config file.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Sequence limit to evaluate. Defaults to max_seq_length from the "
             "config, i.e. what your next training run would actually use.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only check the first N examples (fast smoke test before the "
             "full-dataset scan).",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Measure the valid subset instead of exiting when some labels are "
             "unparseable. The excluded files are always listed.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    target_len = args.max_length if args.max_length is not None else config.get("max_seq_length")

    # Pre-flight the labels before paying for the model load: a single bad JSON
    # file otherwise blows up minutes later with no filename attached.
    pairs = find_label_image_pairs(config)
    invalid = []
    for img_path, label_path in pairs:
        try:
            json.loads(label_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            invalid.append((label_path, e))
    if invalid:
        print(f"\n🚨 {len(invalid)} of {len(pairs)} label file(s) are not valid JSON:")
        for label_path, e in invalid:
            print(f"   {label_path}: {e}")
        if not args.skip_invalid:
            print("\nFix these (they would also crash training), or re-run with "
                  "--skip-invalid to measure the valid subset only.")
            sys.exit(1)
        bad = {p for p, _ in invalid}
        pairs = [(i, l) for i, l in pairs if l not in bad]
        print(f"\n--skip-invalid: measuring the remaining {len(pairs)} example(s). "
              f"The numbers below EXCLUDE the files listed above.")

    model, processor = FastVisionModel.from_pretrained(
        config["base_model"],
        load_in_4bit=config.get("load_in_4bit", True),
        max_seq_length=config.get("max_seq_length", 2048),
    )

    # Same masking config as src/train.py — the response markers must match or
    # the RESPONSE numbers below are meaningless. Only max_seq_length differs.
    collator = UnslothVisionDataCollator(
        model,
        processor,
        # Must match src/train.py, or the image-token counts below describe a
        # resolution training never uses.
        resize=config["image_resize"],
        max_seq_length=NO_TRUNCATION,
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # Limit is applied inside build_conversations so a smoke test doesn't decode
    # every image in the dataset first.
    convos = build_conversations(config, pairs=pairs, limit=args.limit)
    names = [label_path.stem for _, label_path in pairs[: args.limit]]

    image_token_id = getattr(model.config, "image_token_id", None)

    total_lengths = []
    response_lengths = []
    image_lengths = []
    prompt_lengths = []
    overflowing = []              # (name, overflow, response_len) at target_len
    empty_response_examples = []  # examples where NOTHING is unmasked

    for i, convo in enumerate(convos):
        batch = collator([convo])
        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]

        # attention_mask is the only reliable non-pad count: pad_token_id is not
        # exposed consistently on processors, and for some models it collides
        # with a token that legitimately appears in the sequence.
        if "attention_mask" in batch:
            total_len = int(batch["attention_mask"][0].sum())
        else:
            total_len = len(input_ids)

        response_len = int((labels != -100).sum())
        image_len = int((input_ids == image_token_id).sum()) if image_token_id is not None else 0

        total_lengths.append(total_len)
        response_lengths.append(response_len)
        image_lengths.append(image_len)
        prompt_lengths.append(total_len - response_len - image_len)

        if response_len == 0:
            empty_response_examples.append(names[i])

        if target_len is not None:
            # The processor right-truncates, and the response is last, so ANY
            # overflow eats response tokens first.
            overflow = total_len - target_len
            if overflow > 0:
                overflowing.append((names[i], overflow, response_len))

    summarize("TOTAL sequence length (image + prompt + response)", total_lengths)
    summarize("RESPONSE-only length (what actually gets trained on)", response_lengths)
    if image_token_id is not None:
        summarize("IMAGE tokens (shrink with collator resize=, not max_seq_length)", image_lengths)
        summarize("PROMPT text tokens (dead weight, same every example)", prompt_lengths)
    else:
        summarize("PROMPT + IMAGE length (dead weight per example)",
                  [t - r for t, r in zip(total_lengths, response_lengths)])

    if empty_response_examples:
        shown = empty_response_examples[:10]
        more = "..." if len(empty_response_examples) > 10 else ""
        print(f"\n⚠️  {len(empty_response_examples)} example(s) have ZERO unmasked "
              f"response tokens at full length ({', '.join(shown)}{more}). "
              f"That is a template/marker mismatch, not a length issue — check "
              f"instruction_part/response_part against the model's chat template.")

    if target_len is None:
        print("\nNo max_seq_length in config and no --max-length given, so no "
              "truncation check was run.")
    elif overflowing:
        worst = max(overflowing, key=lambda t: t[1])
        fully_lost = [o for o in overflowing if o[1] >= o[2]]
        print(f"\n🚨 At max_seq_length={target_len}, {len(overflowing)} of "
              f"{len(convos)} examples exceed the limit. Right-truncation eats "
              f"the tail = the response, since it's last.")
        print(f"   Worst case: {worst[0]} overflows by {worst[1]} tokens out of "
              f"a {worst[2]}-token response.")
        if fully_lost:
            print(f"   {len(fully_lost)} of them lose their ENTIRE response — "
                  f"those examples would contribute zero gradient signal.")
    else:
        print(f"\n✅ At max_seq_length={target_len}, no example's total length "
              f"exceeds the limit. Responses survive intact.")

    observed_max = max(total_lengths)
    recommended = -(-(observed_max + 64) // 256) * 256  # round up to 256
    print(f"\nMax observed total length: {observed_max}")
    print(f"Recommended max_seq_length (max observed + margin, rounded to 256): {recommended}")
    if args.limit is not None:
        print(f"NOTE: only {len(convos)} example(s) scanned (--limit). Re-run "
              f"without --limit before trusting these numbers.")


if __name__ == "__main__":
    main()
