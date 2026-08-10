# vlm-training-pipeline

Fine-tune a vision-language model (any Unsloth-supported VLM, e.g. Qwen3.5)
on your own document images + JSON labels using LoRA.

Nothing in `src/` is specific to any one document type. Everything that
changes between projects lives in `configs/config.yaml` and
`prompts/extraction_prompt.txt`.

## Quick start

1. **Drop in your data**
   - Put document images in `dataset/images/` (e.g. `doc_0001.jpg`)
   - Put verified ground-truth JSON in `dataset/labels/` (e.g. `doc_0001.json`)
   - Each label file's name (minus extension) must match an image's name.
   - Optional: use `dataset/labels_draft/` to stage raw/unverified model
     output for manual correction. Only files you move into `labels/` are
     ever used for training — `labels_draft/` is never read by the pipeline.

2. **Write your extraction prompt**
   - Edit `prompts/extraction_prompt.txt` with the instructions you want the
     model to follow (schema, format, edge cases, language, etc.). This is
     the same prompt shown to the model during training and at inference.

3. **Edit `configs/config.yaml`**
   - Set `base_model` to the Unsloth model you want to fine-tune (e.g.
     `unsloth/Qwen3.5-4B` for a quick test, `unsloth/Qwen3.5-9B` for the real run).
   - Adjust LoRA/training hyperparameters if needed — sane defaults are
     already set for an 8GB-VRAM GPU.
   - `max_seq_length` covers the **whole sequence** — prompt tokens + image
     tokens + response tokens combined, not input/output counted separately.
     Check your actual longest example before picking a value:
     ```python
     from utils import build_conversations, load_config
     config = load_config("configs/config.yaml")
     convos = build_conversations(config)
     for c in convos:
         text = c["messages"][0]["content"][0]["text"] + c["messages"][1]["content"][0]["text"]
         print(len(tokenizer.encode(text)))
     ```
     Set `max_seq_length` a bit above the longest observed value — sequences
     that exceed it get silently truncated, which can cut off part of your
     JSON label. For dense multi-row tables on a larger GPU (e.g. RunPod),
     `8192` is a reasonable starting point rather than the `2048` default.

4. **Install dependencies**
   ```bash
   uv add unsloth unsloth_zoo torch torchvision pillow pyyaml trl
   ```

5. **Build and inspect the dataset** (sanity check before training)
   ```bash
   uv run python src/build_dataset.py --config configs/config.yaml
   ```

6. **Train**
   ```bash
   uv run python src/train.py --config configs/config.yaml
   ```
   The trained LoRA adapter is saved to `training.output_dir` in your config
   (default: `outputs/adapters/run1`).

   `UnslothVisionDataCollator` is configured with `train_on_responses_only=True`
   (ChatML markers `<|im_start|>user\n` / `<|im_start|>assistant\n`) so loss is
   only computed on the assistant's JSON, not the prompt or image tokens. This
   is **not** the collator's default behavior — see **Verifying loss masking**
   below before trusting any real run.

7. **Run inference with your fine-tuned adapter**
   ```bash
   uv run python src/infer.py --config configs/config.yaml --image path/to/new_doc.jpg
   ```

## Folder structure

```
vlm-training-pipeline/
├── configs/
│   └── config.yaml              # model, paths, hyperparameters
├── prompts/
│   └── extraction_prompt.txt    # your extraction instructions
├── dataset/
│   ├── images/                  # {id}.jpg
│   ├── labels/                  # {id}.json — VERIFIED ground truth only
│   └── labels_draft/            # {id}.json — unverified draft output, never trained on
├── src/
│   ├── utils.py                 # config/prompt loading, dataset construction
│   ├── build_dataset.py         # CLI: build + preview the dataset
│   ├── train.py                 # CLI: LoRA fine-tune
│   └── infer.py                 # CLI: run the fine-tuned adapter on a new image
└── outputs/
    └── adapters/                # trained LoRA adapters land here
```

## Scaling from a mechanics test to a real fine-tune

To prove the pipeline works end-to-end before committing to a big training
run:
1. Use a small model (e.g. `unsloth/Qwen3.5-4B`) in `config.yaml`.
2. Use just 1-2 verified label/image pairs in `dataset/labels/`.
3. Run steps 5-7 above.

Once that works, swap `base_model` to your real target (e.g.
`unsloth/Qwen3.5-9B`), add your full label set to `dataset/labels/`, and
re-run training — no code changes required.

## Verifying loss masking (do this before trusting any run)

Loss masking is the single easiest thing to get silently wrong: if the
collator isn't told to mask the prompt, training "works" — loss goes down,
nothing crashes — but the model is being trained to reproduce the prompt and
predict image tokens too, badly diluting the signal.

Run this after building the model/collator and before a real training run:

```python
# quick_check.py
from utils import load_config, build_conversations
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

config = load_config("configs/config.yaml")
model, tokenizer = FastVisionModel.from_pretrained(config["base_model"], load_in_4bit=True)
collator = UnslothVisionDataCollator(
    model,
    tokenizer,
    train_on_responses_only=True,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
convos = build_conversations(config)
batch = collator([convos[0]])
labels = batch["labels"][0]
unmasked = labels[labels != -100]
print(tokenizer.decode(unmasked))
```

```bash
uv run python quick_check.py
```

The decoded output should show **only** the assistant's JSON — no
`<|im_start|>user`, no extraction prompt text. If you see the full
conversation instead, masking isn't active; double check
`train_on_responses_only=True` and that `instruction_part`/`response_part`
match your model's actual chat template (Qwen uses ChatML, shown above).

## Troubleshooting

### HuggingFace download stalls partway through (Xet backend)
Large model downloads can stall silently using HF's newer "Xet" transfer
backend — the process hangs with no progress, sometimes even after
`HF_HUB_DISABLE_XET=1`. This is a known upstream issue, not a local
network/setup problem. Fix:

```bash
rm -rf ~/.cache/huggingface/hub/models--unsloth--<model-name>
export HF_HUB_DISABLE_XET=1
uv run hf download unsloth/<model-name> \
  --local-dir ~/.cache/huggingface/hub/models--unsloth--<model-name>
```

`hf download` resumes on re-run, so if it stalls, just re-run the same
command again (2-3 times if needed). If it stalls at the same point every
time, throttle concurrency: `export HF_XET_NUM_CONCURRENT_RANGE_GETS=8`.
Also set `HF_TOKEN` — unauthenticated requests are throttled more
aggressively and can look like a stall.

### `pip install --break-system-packages` doesn't work
That flag is for system Python installs only. Since this project uses `uv`,
use `uv add <package>` (adds as a dependency) or `uv pip install <package>`
(installs into the venv without touching `pyproject.toml`) instead.

## Deploying the fine-tuned model

### Ollama
Ollama only runs GGUF files, so the LoRA adapter needs merging and
converting first:

1. Merge the adapter into the base model:
   ```python
   model.save_pretrained_merged("outputs/merged_model", tokenizer, save_method="merged_16bit")
   ```
2. Convert to GGUF with llama.cpp:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp && pip install -r requirements.txt --break-system-packages
   python convert_hf_to_gguf.py ../outputs/merged_model \
     --outfile ../outputs/khatian-model-BF16.gguf --outtype bf16
   ```
3. Create a Modelfile using the **same chat template used in training**
   (ChatML) — a mismatched template is the most common cause of gibberish
   output after conversion:
   ```
   FROM ./khatian-model-BF16.gguf

   TEMPLATE """{{ if .System }}<|im_start|>system
   {{ .System }}<|im_end|>
   {{ end }}<|im_start|>user
   {{ .Prompt }}<|im_end|>
   <|im_start|>assistant
   """

   PARAMETER stop "<|im_start|>"
   PARAMETER stop "<|im_end|>"
   ```
4. Build and run:
   ```bash
   ollama create khatian-extractor -f Modelfile
   ollama run khatian-extractor
   ```

### vLLM
Simpler — vLLM loads the merged HuggingFace-format model directly, no GGUF
conversion needed:
```bash
vllm serve outputs/merged_model --served-model-name khatian-extractor
```
Use the same ChatML template when sending requests.