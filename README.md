# Piper Trainer RU

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/egor125552/piper-trainer-ru/blob/main/Piper_Trainer_RU_Qwen3_ASR.ipynb)

Google Colab notebook for fine-tuning Russian Piper voices.

Features:
- Qwen3-ASR 1.7B: one transcription pass over the complete source in 15–29-second context blocks
- Qwen3-ForcedAligner 0.6B: word timestamps derived from that transcription without ASR again
- only then automatic 3–6-second slicing between words, with non-overlapping WAVs
- VoiceOver-friendly text review
- Piper fine-tuning from ru_RU-dmitri-medium
- configurable checkpoint frequency and retention
- Google Drive checkpoint storage
- ONNX export

## Open in Colab

[Open Piper Trainer RU in Google Colab](https://colab.research.google.com/github/egor125552/piper-trainer-ru/blob/main/Piper_Trainer_RU_Qwen3_ASR.ipynb)

## Notebook

Piper_Trainer_RU_Qwen3_ASR.ipynb

## Прогресс в выходных данных существующих ячеек Colab

В ячейке 5 запускается обычный интерфейс Gradio, но он не встраивается в
вывод Colab: выводится только ссылка. При нажатии кнопки «Распознать и
подготовить датасет» в существующем интерфейсе backend печатает в вывод
этой же ячейки: «Qwen3-ASR: расшифровано N/M; осталось K». Общее число
длинных блоков определяется до распознавания; далее отдельно выводится
прогресс ForcedAligner.

При нажатии «Начать обучение» backend печатает «ЭПОХА N/M: начата /
завершена; шаг S». Бегущая полоса тренера отключена. Эти счётчики также
появляются в обычном журнале обучения на Google Drive.

Никаких дополнительных ячеек запуска ASR и обучения нет. Существующие
датасеты, checkpoints и параметры запуска не меняются.

## Local smoke test with the official Colab Docker image

On a Linux host with Docker:

    ./scripts/test_setup_in_colab_docker.sh

The script extracts the real setup cell from the notebook and runs it inside
us-docker.pkg.dev/colab-images/public/runtime:latest.

## Быстрый запуск

1. Нажмите кнопку **Open in Colab** выше.
2. Войдите в Google-аккаунт, если Colab попросит.
3. Выберите GPU T4: Runtime → Change runtime type → T4 GPU.
4. Запустите ячейки сверху вниз.
5. При подключении Google Drive подтвердите доступ.
6. По ссылке из ячейки 5 откройте Gradio-интерфейс и сначала используйте **«Проверить Qwen на одном аудиофайле»**.
7. Если распознавание работает, загрузите длинные записи или ZIP и нажмите **«Распознать и подготовить датасет»**.
8. Полная расшифровка сохраняется в full_transcript.txt. В dataset/needs_review.csv попадают лишь сомнительные границы и сегменты; править каждую фразу вручную не требуется. При необходимости используйте VoiceOver-редактор.
9. Перейдите во вкладку **«Обучение»**, выберите число дополнительных эпох и частоту checkpoint.
10. После обучения во вкладке **«Экспорт»** получите ONNX + JSON в ZIP.

Checkpoints, подготовленный датасет и экспорт сохраняются в:

MyDrive/PiperTrainer/<имя проекта>/

### Что уже проверено

Подробный технический журнал находится в SMOKE_TEST.md.

Проверены:
- установка Piper в официальном Colab Docker runtime;
- Piper training CLI;
- настоящий Dmitri training checkpoint;
- экспорт настоящего checkpoint в ONNX;
- проверка ONNX через onnx.checker;
- реальный синтез WAV из экспортированного ONNX;
- ZIP, нарезка аудио, metadata.csv, VoiceOver-редактор;
- checkpoint retention;
- запуск Gradio и HTTP 200;
- API Qwen3ASRProcessor без загрузки полных 1.7B весов.

Распознавание всей длинной записи не означает один вызов модели на 23 минуты: на T4 запись сначала последовательно обрабатывается большими блоками, объединённый текст сохраняется, и лишь после окончания всего прохода загружается aligner и создаются короткие WAV.

ASR и aligner выполняются последовательно, чтобы не держать обе модели в VRAM T4 одновременно. Это не гарантирует 100% безошибочную расшифровку: сомнительные фрагменты сохраняются отдельно вместо включения в обучение.

Проверка backend на Linux использует подставные результаты ASR и ForcedAligner. Реальные веса Qwen3-ASR 1.7B, ForcedAligner 0.6B и качество их работы на конкретном исходном аудио проверяются в Colab с GPU.

### Восстановление одного отклонённого длинного блока

После подключения Google Drive в Colab можно отдельно запустить
`python /content/recover_rejected_block_colab.py --project putin --block 40`,
если этот скрипт загружен из папки `scripts` актуального репозитория.
Скрипт берёт существующую полную расшифровку, использует сохранённую
привязку слов при наличии или повторно запускает только ForcedAligner
для указанного блока, а не ASR всей записи. Вначале пытается нарезать
по 3–6 секунд; при невозможности пробует 2,5–7 секунд строго между словами.
Все новые WAV и тексты попадают исключительно в отдельную папку
`recovery_candidates/` на Drive с пометкой `NEEDS_AUDIO_REVIEW`.
Основной `dataset`, `metadata.csv` и checkpoints не меняются.
Если даже резервная нарезка не проходит, исходный блок остаётся отклонённым.
