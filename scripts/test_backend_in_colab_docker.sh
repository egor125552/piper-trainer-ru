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

# Test-only ASR stub.
from pydub.generators import Sine
from pydub import AudioSegment
import shutil,zipfile,time,wave

events=[]
mock_words="Мы проверяем качество новой записи и границы всех слов "*5

def load_asr():
    events.append("load_asr")
    return object()
def transcribe_file(path):
    events.append("asr")
    seconds=len(AudioSegment.from_wav(path))/1000;return ("Мы проверяем качество новой записи и границы всех слов "*max(2,round(seconds/4))).strip()+"."
def unload_asr():
    events.append("unload_asr")
def load_aligner():
    events.append("load_aligner")
    return object()
def align_context(path, text):
    events.append("align")
    length=len(AudioSegment.from_wav(path))/1000
    tokens=text.split()
    step=(length-0.8)/len(tokens)
    return [
        {"text":token,"start_time":0.34+i*step,
         "end_time":0.34+i*step+min(0.25,step*0.75)}
        for i,token in enumerate(tokens)
    ]
def unload_aligner():
    events.append("unload_aligner")

root=Path("/content/piper-colab/work/context_smoke")
shutil.rmtree(root,ignore_errors=True)
root.mkdir(parents=True)
WORK_ROOT=root/"local"
WORK_ROOT.mkdir()
DRIVE_ROOT=root/"drive"
DRIVE_ROOT.mkdir()

# 16 seconds, including real pauses, with enough words for three short phrases.
audio=AudioSegment.silent(duration=250)
for freq in (410,440,470,500):
    audio+=Sine(freq).to_audio_segment(duration=3700).apply_gain(-14)
    audio+=AudioSegment.silent(duration=300)
source=root/"original.wav"
audio.export(source,format="wav")
long_source=root/"long.wav"
(audio*4).export(long_source,format="wav")
zfile=root/"source.zip"
with zipfile.ZipFile(zfile,"w") as archive:
    archive.write(long_source,"recordings/long.wav")

blocks=context_blocks(audio*4)
assert len(blocks)>=2,blocks
assert all(blocks[i][1]==blocks[i+1][0] for i in range(len(blocks)-1))
assert blocks[0][0]==0 and blocks[-1][1]==len(audio*4)

words=align_context(source," ".join(mock_words.split()))
groups=word_groups(words,len(audio)/1000)
ranges=word_group_audio_ranges(groups,len(audio),90)
assert len(groups)>=2,groups
assert all(ranges[i][1]<=ranges[i+1][0] for i in range(len(ranges)-1))
assert all(2800<=b-a<=7100 for a,b in ranges),ranges
assert alignment_matches_transcript(words,mock_words)
assert not alignment_matches_transcript(words[:2],mock_words)
# Equal-length but unrelated words must not pass a count-only alignment check.
assert not alignment_matches_transcript(
    [{"text":"вода"},{"text":"сила"},{"text":"время"}],
    "кошка бегала вчера",
)
# Punctuation and ASR hyphenation changes must not invalidate correct words.
assert alignment_matches_transcript(
    [{"text":"Во"},{"text":"первых"},{"text":"мы"},{"text":"работаем"}],
    "Во-первых, мы работаем.",
)
# ForcedAligner removes punctuation, which must be restored for Piper.
punctuation_groups=[
    [{"text":"большое"},{"text":"значение"}],
    [{"text":"Нам"},{"text":"все"},{"text":"во-первых"}],
]
assert punctuate_aligned_groups(
    punctuation_groups,"большое значение. Нам все, во-первых."
)==["большое значение.","Нам все, во-первых."]
assert punctuate_aligned_groups(
    [[{"text":"ODKB"},{"text":"не"}]],"ODKB не."
)==["ODKB не."]
# When multiple 3-6s segmentations are possible, do not end a phrase
# on a conjunction merely because that boundary is closer to 4.5s.
syntax_words=[
    {"text":("и" if i==9 else "работаем"),
     "start_time":i*0.45,"end_time":(i+1)*0.45}
    for i in range(20)
]
syntax_groups=word_groups(syntax_words,9.0,min_sec=3,max_sec=6)
assert len(syntax_groups)==2,syntax_groups
assert syntax_groups[0][-1]["text"]!="и",syntax_groups

# Real Qwen ForcedAligner output on the user recording included punctuation
# and short function words with start_time == end_time, as well as one
# multi-second word (3.12s). Those are not reasons to discard 24s of speech.
relaxed=[
    {"text":"Мы","start_time":0.25,"end_time":0.62},
    {"text":"уже","start_time":0.66,"end_time":1.13},
    {"text":"работаем","start_time":1.16,"end_time":2.35},
    {"text":"с","start_time":2.35,"end_time":2.35},
    {"text":"другом","start_time":2.36,"end_time":3.33},
    {"text":"продолжительно","start_time":3.35,"end_time":6.47},
    {"text":"говорим","start_time":6.52,"end_time":7.33},
    {"text":"и","start_time":7.33,"end_time":7.33},
    {"text":"проверяем","start_time":7.35,"end_time":8.75},
    {"text":"точность","start_time":8.85,"end_time":9.74},
    {"text":"всей","start_time":9.81,"end_time":10.25},
    {"text":"записи.","start_time":10.4,"end_time":11.54}
]
relaxed_groups=word_groups(relaxed,12.0,min_sec=3.0,max_sec=6.0)
assert len(relaxed_groups)>=2,relaxed_groups
assert [w["text"] for g in relaxed_groups for w in g]==[w["text"] for w in relaxed]
assert all(
    relaxed_groups[i][-1]["end_time"]<=relaxed_groups[i+1][0]["start_time"]
    for i in range(len(relaxed_groups)-1)
)

events.clear()
df,summary,review,editor=prepare_dataset(
    [str(zfile)],"Smoke Voice",250,-35,3.0,6.0,90
)
assert len(df)>=2,(summary,df)
assert "Путин" not in str(source)
assert all(2.8<=float(x)<=7.1 for x in df["duration"])
assert len(set(df["file"]))==len(df)
assert "load_asr" in events and "load_aligner" in events
assert events.index("unload_asr")<events.index("load_aligner")
assert max(i for i,v in enumerate(events) if v=="asr")<min(
    i for i,v in enumerate(events) if v=="align"
)
local,drive,_=project_paths("Smoke Voice")
dataset=local/"dataset"
assert (dataset/"full_transcript.txt").exists()
assert (dataset/"context_blocks.json").exists()
alignment_dir=dataset/"alignment_cache"
assert alignment_dir.is_dir()
cached=list(alignment_dir.glob("block_*.json"))
assert len(cached)==len(blocks),(len(cached),len(blocks))
first_alignment=json.loads(cached[0].read_text(encoding="utf-8"))
assert first_alignment["words"] and first_alignment["transcript"]
assert all("start_time" in w and "end_time" in w for w in first_alignment["words"])
assert (drive/"dataset"/"alignment_cache"/cached[0].name).is_file()
assert len((dataset/"full_transcript.txt").read_text().splitlines())==len(blocks)
assert events.count("asr")==len(blocks)
assert events.count("align")==len(blocks)
assert not (dataset/"context_audio").exists()
assert validate_dataset_for_training(dataset)[0]==len(df)
assert (drive/"dataset"/"full_transcript.txt").exists()
assert (drive/"source"/"source_paths.txt").exists()
assert any((drive/"source").glob("source_*.zip"))
review_df=pd.read_csv(dataset/"review.csv")
for previous,current in zip(review_df.itertuples(),list(review_df.itertuples())[1:]):
    assert float(previous.end_sec)<=float(current.start_sec)+0.02

checked=root/"approved.csv"
checked.write_text(df.iloc[0]["file"]+"|Это проверенная фраза для синтеза.\n",encoding="utf-8")
result,newname=create_verified_project("Smoke Voice","Verified Voice",str(checked))
assert newname=="Verified_Voice" and "Verified_Voice" in result
verified,_,_=project_paths("Verified Voice")
assert validate_dataset_for_training(verified/"dataset")[0]==1
assert len((local/"dataset"/"metadata.csv").read_text().splitlines())==len(df)

# Rebuilding the same project must keep the old dataset as a Drive backup.
prepare_dataset([str(source)],"Smoke Voice",250,-35,3.0,6.0,90)
assert list(drive.glob("dataset_backup_*"))
assert (drive/"dataset"/"full_transcript.txt").exists()

# If aligner fails, preserve the already-saved good dataset and checkpoints.
previous=(drive/"dataset"/"metadata.csv").read_bytes()
def align_context(path, text):
    raise ValueError("test aligner failure")
try:
    prepare_dataset([str(source)],"Smoke Voice",250,-35,3.0,6.0,90)
except RuntimeError:
    pass
else:
    raise AssertionError("missing timestamps unexpectedly allowed training")
assert (drive/"dataset"/"metadata.csv").read_bytes()==previous
# A user can correct metadata.csv on Drive while Colab keeps old labels
# and stale phoneme cache. Training must refresh both without touching the
# project's checkpoints, and it must not start training during this test.
drive_meta=drive/"dataset"/"metadata.csv"
original_text=drive_meta.read_text(encoding="utf-8")
drive_meta.write_text(
    original_text.replace("Мы проверяем", "Мы, проверяем", 1),
    encoding="utf-8",
)
(local/"cache").mkdir(exist_ok=True)
(local/"cache"/"old_phonemes").write_text("stale",encoding="utf-8")
(local/"checkpoints").mkdir(exist_ok=True)
(local/"checkpoints"/"keep.ckpt").write_bytes(b"sentinel")
gen=train_voice("Smoke Voice",1,1,"Каждые N эпох",1,2,1,"Dmitri medium")
header=next(gen)
assert "Проверено реальных фраз" in header
assert (dataset/"metadata.csv").read_bytes()==drive_meta.read_bytes()
assert not (local/"cache"/"old_phonemes").exists()
assert (local/"checkpoints"/"keep.ckpt").read_bytes()==b"sentinel"
gen.close()
# If an old Colab cache misses an audio file, restoring it must not alter
# the current Drive dataset or any training checkpoint.
missing_name=drive_meta.read_text(encoding="utf-8").splitlines()[0].split("|")[0]
(dataset/"wav"/missing_name).unlink()
gen=train_voice("Smoke Voice",1,1,"Каждые N эпох",1,2,1,"Dmitri medium")
next(gen)
assert (dataset/"wav"/missing_name).is_file()
assert (local/"checkpoints"/"keep.ckpt").read_bytes()==b"sentinel"
gen.close()

print("CONTEXT_ASR_ALIGN_SPLIT_SMOKE_OK")

''')
(root / "work").mkdir(exist_ok=True)
(root / "work" / "backend_smoke.py").write_text("\n".join(parts), encoding="utf-8")
PY

docker exec "$CONTAINER" bash -lc 'cd /content/piper-colab && python work/backend_smoke.py'
