# Smoke test status

Verified on Linux using the official Google Colab runtime Docker image:

- image: us-docker.pkg.dev/colab-images/public/runtime:latest
- Python: 3.12.13
- PyTorch: 2.11.0+cu128
- Transformers: 5.15.0
- Gradio: 6.24.0
- Hugging Face Hub: 1.27.0
- NumPy: 2.0.2
- librosa: 0.11.0
- soundfile: 0.14.0
- Lightning after Piper train install: 2.6.6
- scikit-build: 0.19.1

System packages confirmed:
- ffmpeg
- gcc / g++
- cmake
- git
- espeak-ng 1.50
- ninja 1.10.1

Piper source commit pinned by the notebook:
5b355b110aecf3de8f4e000ede1ce06831acff35

Verified successfully inside the official Colab Docker runtime:
- Piper training dependencies install
- monotonic_align build
- setup.py build_ext --inplace
- python -m piper.train fit --help
- python -m piper.train.export_onnx --help

Qwen3-ASR processor check:
- Qwen/Qwen3-ASR-1.7B-hf
- processor class: Qwen3ASRProcessor
- apply_transcription_request is available
- local audio paths are supported by apply_transcription_request
- feature extractor sampling rate: 16000 Hz

Public GitHub-backed Colab URL opens successfully and detects a T4 runtime.
Running cells in Colab requires signing in to a Google Account, so the first setup cell could not be executed in the unauthenticated browser smoke test.

Canonical notebook:
Piper_Trainer_RU_Qwen3_ASR.ipynb

Canonical Linux path:
/opt/piper-colab/Piper_Trainer_RU_Qwen3_ASR.ipynb

## Exact notebook setup-cell test

The exact first setup/install cell from Piper_Trainer_RU_Qwen3_ASR.ipynb was executed inside the official Colab Docker runtime.

Final verified result:
- exit code: 0
- Piper training CLI: OK
- Piper ONNX export CLI: OK
- Piper commit pin: 5b355b110aecf3de8f4e000ede1ce06831acff35
- setuptools pinned below 82 for compatibility with Torch 2.11
- jedi installed to satisfy the bundled IPython dependency

The second full setup-cell run completed successfully without the earlier setuptools/IPython dependency conflicts.
