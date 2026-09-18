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
