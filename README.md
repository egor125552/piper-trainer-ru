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

## Прогресс именно в выходных данных Colab

Откройте актуальную версию существующего блокнота из GitHub и выполните
установку и подключение Google Drive. В ячейке 5 Gradio **не встраивается**
в вывод: показывается только ссылка для тех, кому нужны веб-элементы управления.
Для запуска прямо из Colab без перехода в Gradio теперь есть два отдельных
блока в конце того же блокнота.

**Ячейка 6: Qwen3-ASR и подготовка датасета.** Впишите PROJECT_FOR_ASR
и SOURCE_AUDIO_ON_DRIVE (полный путь к записи на подключённом Drive),
отметьте RUN_ASR и выполните ячейку. До загрузки Qwen создаются
длинные WAV-блоки, поэтому точное общее число известно заранее.
После каждого блока прямо под ячейкой печатается «расшифровано N/M;
осталось K». После ASR отдельно показывается прогресс ForcedAligner.
Существующий датасет не перестраивается, пока не отмечено
ALLOW_REBUILD_EXISTING_DATASET.

**Ячейка 7: обучение готового датасета.** Впишите
PROJECT_FOR_TRAINING, число дополнительных эпох и параметры сохранения,
выберите старт от Dmitri или последнего checkpoint на Drive, отметьте
RUN_TRAINING и выполните ячейку. В её выводе появляются отдельные
строки «ЭПОХА N/M: начата / завершена; шаг S», без бегущей
текстовой полоски прогресса. Датасет повторно не готовится.
Для продолжения от предыдущей модели явно выберите
«Последний checkpoint с Google Drive», иначе будет старт от Dmitri.

**Важно:** новые блоки ничего не запускают при RUN_ASR=False и
RUN_TRAINING=False. Не запускайте подготовку для уже готового проекта
putin или my_voice с разрешением перестройки, если задача состоит
только в продолжении обучения. Кнопки существующего интерфейса Gradio
остались рабочими и тоже печатают счётчики в Colab, но для
предсказуемого вывода под собственной ячейкой используйте ячейки 6 и 7.

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
