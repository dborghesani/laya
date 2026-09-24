# Fine-tuning Laya

This guide runs `finetuning.py` from the repository root. It uses Python 3.10 and
PyTorch's CUDA 12.8 build (`cu128`) in both local and cloud environments so the
same lockfile and runtime are used everywhere.

## Requirements

- NVIDIA GPU and an NVIDIA driver compatible with CUDA 12.8.
- Python 3.10.
- [uv](https://docs.astral.sh/uv/).
- Network access to Hugging Face on the first run, to download the base model.

The CUDA toolkit does not need to be installed in the virtual environment.
PyTorch wheels bundle their CUDA user-space libraries; the host only needs a
compatible NVIDIA driver.

## Create the environment

From the repository root:

```bash
uv python install 3.10
uv venv --python 3.10 .venv
uv sync
```

`pyproject.toml` pins `torch==2.11.0+cu128`, so `uv sync` installs the CUDA 12.8
PyTorch build. Do not replace it with a CPU wheel or a wheel targeting a different
CUDA version if local and cloud reproducibility is required.

Verify the environment before training:

```bash
uv run python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

The expected `PyTorch CUDA` value is `12.8` and `CUDA available` should be `True`.

## Run fine-tuning

The default dataset is `dataset/laya_automotive_typed_decisions_50k`, which must
contain `train.jsonl` and `test.jsonl` (or another format supported by the script).

```bash
uv run python scripts/finetuning.py \
  --dataset dataset/laya_automotive_typed_decisions_50k \
  --output output
```

The base model defaults to `convaiinnovations/laya`. Each run is written below
`output/` with a UTC timestamp and process ID. The last model checkpoint is saved
at `output/<run>/checkpoints/last/`; the final calibrated model and benchmark report
are written to `output/<run>/` when evaluation is enabled.

For a short smoke test:

```bash
uv run python scripts/finetuning.py \
  --limit 20 \
  --epochs 1 \
  --batch-size 2 \
  --grad-accum 1
```

## Preprocessing cache

Tokenized training and validation decisions are cached by default in:

```text
dataset/laya_automotive_typed_decisions_50k/laya-preprocessing-cache/
```

The cache is split-specific and automatically changes when the dataset files,
selected tokenizer, `--limit`, `--max-len`, or `--head-max-len` change. To use a
shared persistent cache on a cloud volume:

```bash
uv run python scripts/finetuning.py \
  --preprocessing-cache-dir /mnt/laya-cache
```

To force rebuilding it for one run:

```bash
uv run python scripts/finetuning.py --no-preprocessing-cache
```

## Common modes

Disable post-training calibration and benchmark evaluation while iterating:

```bash
uv run python scripts/finetuning.py --no-evaluate
```

Keep the validation pass but avoid writing the optional best checkpoint:

```bash
uv run python scripts/finetuning.py --no-save-best
```

Evaluate an existing fine-tuned model without training:

```bash
uv run python scripts/finetuning.py \
  --eval-model-dir output/<run> \
  --report output/<run>/benchmark_report.json
```

TensorBoard logs are enabled by default. Launch it from the same environment:

```bash
uv run tensorboard --logdir output
```

## Local/cloud parity

Use the same repository revision, `uv.lock`, Python 3.10, and `uv sync` locally
and in the cloud. This gives both environments the same package versions and the
same CUDA 12.8 PyTorch build. GPU models may differ, but the training code and
runtime dependencies remain aligned.
