#!/usr/bin/env python3
"""Validate the new notebook progress plumbing without GPU or user Drive."""
import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

root=Path(__file__).resolve().parents[1]
nb=json.loads((root/'Piper_Trainer_RU_Qwen3_ASR.ipynb').read_text())
assert len(nb['cells'])==7
code=[''.join(c['source']) for c in nb['cells']]
for i in range(3,7): ast.parse(code[i],filename=f'cell{i}')
assert 'Qwen3-ASR: расшифровано {completed}/{total_blocks}' in code[4]
assert 'осталось {total_blocks-completed}' in code[4]
assert 'Привязка слов: блок {aligned_count}/{len(records)}' in code[4]
assert 'EpochConsoleProgress()' in code[5]
assert '"--trainer.enable_progress_bar"' in code[5]
assert 'inline=False' in code[6] and 'quiet=True' in code[6]
assert not any('RUN_ASR =' in cell or 'RUN_TRAINING =' in cell for cell in code)
assert 'prepare_btn.click(' in code[6] and 'train_btn.click(' in code[6]
assert "report_cell_progress(line)" in code[5]

# Inspect the real runner created by the existing backend smoke; check the
# nested f-string was expanded and its callback prints correct epoch numbers.
runner=root/'work/context_smoke/local/Smoke_Voice/run_piper_training.py'
assert runner.is_file()
tree=ast.parse(runner.read_text())
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='EpochConsoleProgress')
assert {f.name for f in cls.body if isinstance(f,ast.FunctionDef)}=={'on_train_epoch_start','on_train_epoch_end'}
class Callback: pass
env={'Callback':Callback}
exec(compile(ast.Module(body=[cls],type_ignores=[]),str(runner),'exec'),env)
cb=env['EpochConsoleProgress']()
fake=SimpleNamespace(current_epoch=49,max_epochs=100,global_step=1200)
out=io.StringIO()
with contextlib.redirect_stdout(out):
    cb.on_train_epoch_start(fake,None)
    cb.on_train_epoch_end(fake,None)
s=out.getvalue()
assert 'ЭПОХА 50/100: начата; шаг 1200.' in s,s
assert 'ЭПОХА 50/100: завершена; шаг 1200.' in s,s
print('COLAB_PROGRESS_SMOKE_OK')
