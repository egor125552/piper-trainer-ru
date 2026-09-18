#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${COLAB_CONTAINER:-piper-colab-runtime}"

python3 - "$ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
nb = json.loads((root / "Piper_Trainer_RU_Qwen3_ASR.ipynb").read_text(encoding="utf-8"))
parts = [
    "import sys\n",
    "from pathlib import Path\n",
    "DRIVE_ROOT=Path('/content/piper-colab/work/fake_drive')\n",
    "DRIVE_ROOT.mkdir(parents=True,exist_ok=True)\n",
    "DMITRI_CKPT='/content/piper-colab/work/fake_base.ckpt'\n",
    "Path(DMITRI_CKPT).write_bytes(b'fake')\n",
]
for i in (3, 4, 5):
    parts.append("".join(nb["cells"][i]["source"]))
parts.append(r'''
def load_asr():
    return object()
def transcribe_file(path):
    return "Тестовая фраза для проверки датасета."
def unload_asr():
    pass

from pydub.generators import Sine
from pydub import AudioSegment
import zipfile, shutil, os, time

root=Path('/content/piper-colab/work/backend_smoke')
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True)
WORK_ROOT=root/'local'
WORK_ROOT.mkdir()

audio = Sine(440).to_audio_segment(duration=2000).apply_gain(-12)
audio += AudioSegment.silent(duration=800)
audio += Sine(660).to_audio_segment(duration=2000).apply_gain(-12)
src=root/'long.wav'
audio.export(src,format='wav')

zpath=root/'input.zip'
with zipfile.ZipFile(zpath,'w') as z:
    z.write(src,'nested/long.wav')
    z.writestr('__MACOSX/junk.txt','junk')

df, summary, review, editor = prepare_dataset(
    [str(zpath)], 'Smoke Voice',
    min_silence_ms=400,
    silence_db=-35,
    min_sec=1.0,
    max_sec=12,
    padding_ms=100,
)
assert len(df)==2
metadata=WORK_ROOT/safe_name('Smoke Voice')/'dataset'/'metadata.csv'
lines=metadata.read_text(encoding='utf-8').splitlines()
assert len(lines)==2

edited='\n'.join([
    lines[0].split('|',1)[0]+'|Первая исправленная фраза.',
    lines[1].split('|',1)[0]+'|Вторая исправленная фраза.',
])
apply_text_review('Smoke Voice',edited)

local, drive_p, _=project_paths('Smoke Voice')
checked, minutes = validate_dataset_for_training(local/'dataset')
assert checked == 2
assert minutes > 0
assert phrase_quality_reason('Это незаконченная фраза...', 2.5)
continuous = Sine(440).to_audio_segment(duration=2500).apply_gain(-12)
assert phrase_boundary_reason(continuous, 200, 2200, 100)

bad = lines[0].split('|',1)[0]+'|Оборванная фраза без точки\n'+edited.splitlines()[1]
(local/'dataset'/'metadata.csv').write_text(bad, encoding='utf-8')
try:
    validate_dataset_for_training(local/'dataset')
except ValueError as exc:
    assert 'сомнительные фразы' in str(exc)
else:
    raise AssertionError('bad metadata unexpectedly passed validation')
apply_text_review('Smoke Voice',edited)

shutil.rmtree(local/'dataset')
restore_dataset_from_drive('Smoke Voice')

lc=local/'checkpoints'
dc=drive_p/'checkpoints'
lc.mkdir(parents=True,exist_ok=True)
for i in range(5):
    p=lc/f'test-{i}.ckpt'
    p.write_bytes(bytes([i])*10)
    os.utime(p,(time.time()+i,time.time()+i))
sync_checkpoints(lc,dc,2)
assert len(list(lc.glob('*.ckpt')))==2
assert len(list(dc.glob('*.ckpt')))==2

print('BACKEND_SMOKE_TEST_OK')
''')
(root / "work").mkdir(exist_ok=True)
(root / "work" / "backend_smoke.py").write_text("\n".join(parts), encoding="utf-8")
PY

docker exec "$CONTAINER" bash -lc 'cd /content/piper-colab && python work/backend_smoke.py'
