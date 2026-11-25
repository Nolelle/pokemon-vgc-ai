#!/usr/bin/env bash
# Install the locked dev environment plus the `train` extra's packages, with torch taken
# from PyTorch's CPU index. A plain `uv sync --extra train` on Linux pulls the CUDA build
# (~3 GB of nvidia-* wheels) that CI never uses. Every version, including torch's and
# all other `train` dependencies, is constrained to uv.lock. Linux CI only: PyTorch
# publishes no `+cpu` wheel for macOS, where the plain PyPI torch is already CPU/MPS.
set -euo pipefail

uv sync --frozen --extra dev

constraints="$(mktemp)"
train_reqs="$(mktemp)"
trap 'rm -f "$constraints" "$train_reqs"' EXIT

uv export --frozen --extra dev --extra train --no-hashes --no-emit-project --no-header \
  | grep -vE '^(nvidia-|triton)' > "$constraints"
torch_version="$(grep -E '^torch==' "$constraints" | cut -d'=' -f3 | cut -d' ' -f1 | cut -d';' -f1)"

# Derive non-torch `train` packages dynamically from pyproject.toml so adding a new
# training dependency never gets silently omitted in CI.
.venv/bin/python -c '
import re, tomllib
with open("pyproject.toml", "rb") as f:
    reqs = tomllib.load(f)["project"]["optional-dependencies"]["train"]
for req in reqs:
    spec = re.split(r"[<>=!~;\s]", req, maxsplit=1)[0].strip()
    base = spec.split("[", 1)[0].lower()
    if spec and base != "torch":
        print(spec)
' > "$train_reqs"

install_args=("torch==${torch_version}+cpu")
if [ -s "$train_reqs" ]; then
  install_args+=(-r "$train_reqs")
fi

# torch is pinned to the `+cpu` build, which only the PyTorch index has; everything else
# resolves from PyPI at the locked version (the PyTorch index lacks e.g. setuptools 83).
uv pip install --python .venv/bin/python -c "$constraints" \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  "${install_args[@]}"
