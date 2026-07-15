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
