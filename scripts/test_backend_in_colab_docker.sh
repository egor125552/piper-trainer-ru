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
checked_file=root/'metadata_checked.csv'
checked_file.write_text(edited, encoding='utf-8')
# UI and backend smoke share a persistent fake Drive container.
verified_previous=DRIVE_ROOT/'Smoke_Verified'
if verified_previous.exists():
    shutil.rmtree(verified_previous)
checked_result,new_project=create_verified_project('Smoke Voice','Smoke Verified',str(checked_file))
assert 'Smoke_Verified' in checked_result and new_project=='Smoke_Verified'
verified_local, verified_drive, _=project_paths('Smoke Verified')
assert len((verified_local/'dataset'/'metadata.csv').read_text().splitlines())==2
assert (verified_drive/'dataset'/'approved_by_review.txt').exists()
try:
    create_verified_project('Smoke Voice','Smoke Verified',str(checked_file))
except FileExistsError:
    pass
else:
    raise AssertionError('verified import overwrote an existing dataset')
invalid_file=root/'invalid_review.csv'
invalid_file.write_text('unknown.wav|Несуществующая запись.\n', encoding='utf-8')
try:
    create_verified_project('Smoke Voice','Smoke Untrusted',str(invalid_file))
except ValueError:
    pass
else:
    raise AssertionError('unknown file slipped through manual verification')
original_cache = list((drive_p/'source').glob('source_*.zip'))
assert original_cache
source_paths=(drive_p/'source'/'source_paths.txt').read_text(encoding='utf-8').splitlines()
assert len(source_paths)==1 and Path(source_paths[0]).exists()
assert Path(source_paths[0]).read_bytes()==zpath.read_bytes()
overlap_csv=local/'dataset'/'review.csv'
original_review=overlap_csv.read_text(encoding='utf-8')
overlap_table=pd.read_csv(overlap_csv)
overlap_table.loc[1,'start_sec']=float(overlap_table.loc[0,'end_sec'])-0.3
overlap_table.to_csv(overlap_csv,index=False,encoding='utf-8')
try:
    validate_dataset_for_training(local/'dataset')
except ValueError as exc:
    assert 'перекрытие речи' in str(exc)
else:
    raise AssertionError('overlapping clips should not pass the training gate')
overlap_csv.write_text(original_review,encoding='utf-8')
assert phrase_quality_reason('Это незаконченная фраза...', 2.5)
assert phrase_quality_reason('Мы начинаем проверку без конечного знака', 3.0, allow_unpunctuated_clause=True) is None
assert unfinished_syntax('Мы должны быть готовы к')
assert phrase_quality_reason('Мы должны быть готовы к', 3.0, allow_unpunctuated_clause=True)
assert not unfinished_syntax('Мы закончили важную совместную работу')
assert not phrase_quality_reason('Это проверенная фраза.', 2.5)
assert phrase_quality_reason('Мы всё обсудили, но.', 2.5)
# Previously 350 ms padding duplicated neighboring speech across their WAVs.
adjacent = (
    AudioSegment.silent(duration=400)
    + Sine(440).to_audio_segment(duration=1800).apply_gain(-12)
    + AudioSegment.silent(duration=220)
    + Sine(660).to_audio_segment(duration=1800).apply_gain(-12)
    + AudioSegment.silent(duration=400)
)
ranges = smart_segment_ranges(
    adjacent, min_silence_ms=160, silence_db=-35,
    min_sec=1.0, max_sec=3.0, padding_ms=350, merge_gap_ms=0,
)
assert len(ranges) == 2, ranges
assert ranges[0][1] <= ranges[1][0], ranges
quiet = AudioSegment.silent(duration=400) + Sine(440).to_audio_segment(duration=2000).apply_gain(-12) + AudioSegment.silent(duration=400)
assert natural_clause_boundary(quiet, 200, 2600)
assert not natural_clause_boundary(Sine(440).to_audio_segment(duration=2500).apply_gain(-12), 200, 2200)

original_transcribe = transcribe_file
def transcribe_file(path):
    if '000002' in Path(path).stem:
        return 'Мы должны быть готовы к'
    return 'Мы выполнили полезную проверку.'
filtered, _, _, _ = prepare_dataset([str(src)], 'Review Voice', 400, -35, 1.0, 12, 100)
assert len(filtered)==1
review_local, _, _ = project_paths('Review Voice')
needs_review = review_local/'dataset'/'needs_review.csv'
assert needs_review.exists()
assert 'needs_review_wav' in needs_review.read_text(encoding='utf-8')
assert len(list((review_local/'dataset'/'needs_review_wav').glob('*.wav')))==1
transcribe_file = original_transcribe
clause_src=root/'clauses.wav'
quiet.export(clause_src, format='wav')
def transcribe_file(path):
    return 'Здесь звучит обычная фраза'
clauses, _, _, _ = prepare_dataset([str(clause_src)], 'Clause Voice', 200, -35, 1.0, 12, 350)
assert len(clauses)==1, len(clauses)
clause_local, _, _ = project_paths('Clause Voice')
clause_files=(clause_local/'dataset'/'natural_clause_files.txt').read_text(encoding='utf-8').splitlines()
assert len(clause_files)==1
assert validate_dataset_for_training(clause_local/'dataset')[0]==1
transcribe_file = original_transcribe

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
