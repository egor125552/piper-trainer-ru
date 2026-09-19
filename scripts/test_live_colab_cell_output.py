#!/usr/bin/env python3
"""Exercise actual notebook launch-cell tail with fake Gradio and worker events."""
import ast
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
nb = json.loads((root / "Piper_Trainer_RU_Qwen3_ASR.ipynb").read_text())
assert len(nb["cells"]) == 7
backend = ast.parse("".join(nb["cells"][3]["source"]))
helper = next(
    node for node in backend.body
    if isinstance(node, ast.FunctionDef) and node.name == "report_cell_progress"
)
launch = "".join(nb["cells"][6]["source"])
launch = launch[launch.index("# This ORIGINAL launch cell"):]
assert "while True:" in launch
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "colab_cell_progress.log"
    ctx = {
        "CELL_PROGRESS_FILE": path,
        "_CELL_PROGRESS_LOCK": threading.Lock(),
    }
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "<report-helper>", "exec"), ctx)
    source = """
import sys
from pathlib import Path
CELL_PROGRESS_FILE = Path(sys.argv[1])
class Demo:
    def queue(self, **kwargs):
        return self
    def launch(self, **kwargs):
        assert kwargs["inline"] is False
        return (None, "http://127.0.0.1:7860/", None)
demo = Demo()
""" + launch
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", source, str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    observed = [""]
    def until(phrase, seconds=8):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if phrase in observed[0]:
                return
            if proc.poll() is not None:
                raise AssertionError(f"Cell terminated unexpectedly: {observed[0]}")
            readable, _, _ = select.select([proc.stdout], [], [], 0.2)
            if readable:
                observed[0] += os.read(proc.stdout.fileno(), 4096).decode("utf-8")
        raise AssertionError(f"No output containing {phrase!r}: {observed[0]}")
    try:
        until("Ячейка продолжает работать")
        assert proc.poll() is None, "Launch cell exited right after link"
        ctx["report_cell_progress"]("Qwen3-ASR: расшифровано 3/12; осталось 9.")
        until("расшифровано 3/12; осталось 9")
        ctx["report_cell_progress"]("ЭПОХА 4/100: начата; шаг 72.")
        until("ЭПОХА 4/100: начата; шаг 72")
        assert proc.poll() is None, "Launch cell exited after showing progress"
        print("LIVE_COLAB_CELL_OUTPUT_OK")
        print(observed[0].strip())
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        proc.communicate(timeout=4)
