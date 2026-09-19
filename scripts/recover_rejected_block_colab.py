#!/usr/bin/env python3
"""Recover only one rejected Qwen context block. Never change the main dataset."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

def recover(record, words, audio, folder, backend):
    if not backend["alignment_matches_transcript"](words, record["text"]):
        raise ValueError("Слова ForcedAligner не совпадают с расшифровкой")
    mode="обычный"
    try:
        groups=backend["word_groups"](words,len(audio)/1000,min_sec=3.0,max_sec=6.0)
    except ValueError:
        groups=backend["word_groups"](words,len(audio)/1000,min_sec=2.5,max_sec=7.0)
        mode="резервный 2.5–7 с"
    spans=backend["word_group_audio_ranges"](groups,len(audio),pad_ms=90)
    texts=backend["punctuate_aligned_groups"](groups,record["text"])
    if len(groups)!=len(spans) or len(groups)!=len(texts):
        raise ValueError("Число фраз не совпало")
    if folder.exists():
        raise FileExistsError(f"Папка кандидатов уже существует: {folder}")
    folder.mkdir(parents=True)
    (folder/"wav").mkdir()
    (folder/"aligned_words.json").write_text(
        json.dumps({"record":record,"words":words},ensure_ascii=False,indent=2),
        encoding="utf-8"
    )
    rows=[]
    for i,(group,(begin,end),text) in enumerate(zip(groups,spans,texts),1):
        if not (0<=begin<end<=len(audio)) or len(group)<2 or not text:
            raise ValueError(f"Недопустимая фраза {i}")
        if rows and begin<rows[-1]["relative_end_ms"]:
            raise ValueError(f"Перекрытие фраз {i}")
        name=f"block_{record['block_index']:03d}_{i:02d}.wav"
        audio[begin:end].export(folder/"wav"/name,format="wav")
        rows.append({
            "file":name,"text":text,
            "duration_sec":round((end-begin)/1000,3),
            "source_start_sec":round((record["start_ms"]+begin)/1000,3),
            "source_end_sec":round((record["start_ms"]+end)/1000,3),
            "relative_start_ms":begin,"relative_end_ms":end,
            "status":"NEEDS_AUDIO_REVIEW","mode":mode
        })
    with (folder/"review_candidates.csv").open("w",encoding="utf-8",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (folder/"README.txt").write_text(
        "Фразы сохранены только для проверки, не добавлены в датасет.\n"
        "Проверьте каждый WAV и его текст на слух, включая границы.\n"
        f"Режим: {mode}; фраз: {len(rows)}.\n",encoding="utf-8"
    )
    return rows,mode

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--project",default="putin")
    parser.add_argument("--block",type=int,default=40)
    parser.add_argument("--drive-root",default="/content/drive/MyDrive/PiperTrainer")
    parser.add_argument("--notebook",default="/content/Piper_Trainer_RU_Qwen3_ASR.ipynb")
    args=parser.parse_args()
    root=Path(args.drive_root)
    project=root/args.project
    blocks=json.loads((project/"dataset/context_blocks.json").read_text(encoding="utf-8"))
    records=[r for r in blocks if int(r["block_index"])==args.block]
    if len(records)!=1:
        raise ValueError(f"Блок {args.block} не найден")
    record=records[0]
    sources=list((project/"source").glob("source_*.*"))
    sources=[p for p in sources if p.suffix.lower() in (".mp3",".wav",".m4a",".flac")]
    if len(sources)!=1:
        raise ValueError(f"Нужно ровно одно исходное аудио, найдено: {len(sources)}")
    from pydub import AudioSegment
    original=AudioSegment.from_file(sources[0])
    audio=original[record["start_ms"]:record["end_ms"]].set_channels(1).set_frame_rate(22050)
    if abs(len(audio)-(record["end_ms"]-record["start_ms"]))>100:
        raise ValueError("Длительность исходного блока не совпала")
    cache=project/"dataset/alignment_cache"/f"block_{args.block:03d}.json"
    namespace=None
    try:
        notebook=json.loads(Path(args.notebook).read_text(encoding="utf-8"))
        namespace={"sys":sys,"__name__":"recover_context","DRIVE_ROOT":root,"DMITRI_CKPT":""}
        for idx in (3,4):
            exec(compile("".join(notebook["cells"][idx]["source"]),f"notebook_cell_{idx}","exec"),namespace)
        if cache.exists():
            cached=json.loads(cache.read_text(encoding="utf-8"))
            if (cached["start_ms"],cached["end_ms"],cached["transcript"]) != (
                record["start_ms"],record["end_ms"],record["text"]
            ):
                raise ValueError("Кэш привязки не относится к этому блоку")
            words=cached["words"]
            print("USING_CACHED_ALIGNMENT",flush=True)
        else:
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                audio_file=Path(tmp)/"context.wav"
                audio.export(audio_file,format="wav")
                print("ALIGNING_ONE_BLOCK",args.block,flush=True)
                alignment=namespace["align_context"](str(audio_file),record["text"])
            words=[{"text":str(w["text"]),"start_time":float(w["start_time"]),
                    "end_time":float(w["end_time"])} for w in alignment]
        folder=project/"recovery_candidates"/(
            f"block_{args.block:03d}_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        )
        rows,mode=recover(record,words,audio,folder,namespace)
        print("RECOVERY_REVIEW_CANDIDATES",len(rows),"MODE",mode,flush=True)
        print("CANDIDATES_FOLDER",folder,flush=True)
    finally:
        if namespace is not None:
            namespace["unload_aligner"]()

if __name__=="__main__":
    main()
