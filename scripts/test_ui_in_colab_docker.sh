#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${COLAB_CONTAINER:-piper-colab-runtime}"
PORT="${UI_SMOKE_PORT:-17860}"

python3 - "$ROOT" "$PORT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
port = int(sys.argv[2])
nb = json.loads((root / "Piper_Trainer_RU_Qwen3_ASR.ipynb").read_text(encoding="utf-8"))
base = (root / "work" / "backend_smoke.py").read_text(encoding="utf-8").split("# Test-only ASR stub.")[0]
ui = "".join(nb["cells"][6]["source"])
ui = ui.replace(
    'demo.queue(default_concurrency_limit=1).launch(share=True, debug=True)',
    f'''demo.queue(default_concurrency_limit=1).launch(
    server_name="0.0.0.0",
    server_port={port},
    share=False,
    prevent_thread_lock=True,
)
import time
time.sleep(30)
'''
)
(root / "work" / "ui_server_smoke.py").write_text(base + "\n" + ui, encoding="utf-8")
PY

docker exec -d "$CONTAINER" bash -lc 'cd /content/piper-colab && python work/ui_server_smoke.py >/tmp/ui_server_smoke.log 2>&1'

for _ in $(seq 1 20); do
  if docker exec "$CONTAINER" bash -lc "python - <<PY
import requests
r=requests.get('http://127.0.0.1:$PORT', timeout=2)
print(r.status_code, len(r.content))
assert r.status_code == 200
PY" >/tmp/ui_http.out 2>/dev/null; then
    cat /tmp/ui_http.out
    echo UI_HTTP_OK
    docker exec "$CONTAINER" pkill -f 'python work/ui_server_smoke.py' || true
    exit 0
  fi
  sleep 2
done

docker exec "$CONTAINER" tail -120 /tmp/ui_server_smoke.log || true
exit 1
