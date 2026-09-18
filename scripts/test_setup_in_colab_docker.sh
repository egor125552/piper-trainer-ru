#!/usr/bin/env bash
set -euo pipefail

IMAGE="${COLAB_IMAGE:-us-docker.pkg.dev/colab-images/public/runtime:latest}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$ROOT/work"
mkdir -p "$WORK"

python3 - "$ROOT" "$WORK" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
work = pathlib.Path(sys.argv[2])
nb = json.loads((root / "Piper_Trainer_RU_Qwen3_ASR.ipynb").read_text(encoding="utf-8"))
code = "".join(nb["cells"][1]["source"])
out = work / "setup_cell.py"
out.write_text(code, encoding="utf-8")
print(f"Extracted setup cell -> {out}")
PY

echo "Using image: $IMAGE"
docker run --rm \
  --entrypoint bash \
  -v "$ROOT:/content/piper-colab" \
  "$IMAGE" \
  -lc 'cd /content/piper-colab && python work/setup_cell.py'

echo "SETUP_SMOKE_TEST_OK"
