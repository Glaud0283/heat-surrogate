#!/usr/bin/env bash
# Download the model release pinned in model.lock and verify its checksum.
# Works the same on your laptop (after `gh auth login`) and in CI (with GH_TOKEN set).
set -euo pipefail

source model.lock                       # sets MODEL_VERSION and MODEL_SHA256
mkdir -p models
gh release download "$MODEL_VERSION" --pattern model.pt --dir models --clobber
echo "$MODEL_SHA256  models/model.pt" | sha256sum --check -

# In GitHub Actions, also export MODEL_VERSION to every later step of this job.
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "MODEL_VERSION=$MODEL_VERSION" >> "$GITHUB_ENV"
fi
