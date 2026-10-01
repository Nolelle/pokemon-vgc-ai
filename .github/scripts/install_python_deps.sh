#!/usr/bin/env bash
# Install the locked dev environment plus the `train` extra's packages, with torch taken
# from PyTorch's CPU index. A plain `uv sync --extra train` on Linux pulls the CUDA build
# (~3 GB of nvidia-* wheels) that CI never uses. Every version, including torch's and
# wandb's own dependencies, is constrained to uv.lock. Linux CI only: PyTorch publishes
# no `+cpu` wheel for macOS, where the plain PyPI torch is already CPU/MPS.
set -euo pipefail

uv sync --frozen --extra dev

constraints="$(mktemp)"
uv export --frozen --extra dev --extra train --no-hashes --no-emit-project --no-header \
  | grep -vE '^(nvidia-|triton)' > "$constraints"
torch_version="$(grep -E '^torch==' "$constraints" | cut -d'=' -f3 | cut -d' ' -f1)"

# torch is pinned to the `+cpu` build, which only the PyTorch index has; everything else
# resolves from PyPI at the locked version (the PyTorch index lacks e.g. setuptools 83).
uv pip install --python .venv/bin/python -c "$constraints" \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  "torch==${torch_version}+cpu" wandb
