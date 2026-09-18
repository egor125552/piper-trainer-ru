# Piper Trainer RU

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/egor125552/piper-trainer-ru/blob/main/Piper_Trainer_RU_Qwen3_ASR.ipynb)

Google Colab notebook for fine-tuning Russian Piper voices.

Features:
- Qwen3-ASR 1.7B transcription
- automatic slicing of long recordings
- VoiceOver-friendly text review
- Piper fine-tuning from ru_RU-dmitri-medium
- configurable checkpoint frequency and retention
- Google Drive checkpoint storage
- ONNX export

## Open in Colab

[Open Piper Trainer RU in Google Colab](https://colab.research.google.com/github/egor125552/piper-trainer-ru/blob/main/Piper_Trainer_RU_Qwen3_ASR.ipynb)

## Notebook

Piper_Trainer_RU_Qwen3_ASR.ipynb

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
6. В открывшемся Gradio-интерфейсе сначала используйте **«Проверить Qwen на одном аудиофайле»**.
7. Если распознавание работает, загрузите длинные записи или ZIP и нажмите **«Распознать и подготовить датасет»**.
8. Исправьте текст в VoiceOver-friendly редакторе: имя.wav|текст.
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

Полный Qwen3-ASR 1.7B и реальный training step требуют GPU и проверяются уже в облачном Colab T4.
