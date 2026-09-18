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

## Backend smoke test

Verified inside the official Colab Docker runtime with ASR stubbed:
- ZIP audio input extraction
- silence-based slicing
- generation of two WAV clips
- metadata.csv creation
- VoiceOver text-editor path
- restoring dataset from the Drive-like mirror
- checkpoint copy and retention policy

Result: BACKEND_SMOKE_TEST_OK

## Gradio smoke test

The exact UI cell was instantiated with Gradio 6.24 and then launched locally
inside the official Colab Docker runtime. HTTP GET to the local UI returned 200.

Result: UI_HTTP_OK

## Real Dmitri checkpoint and ONNX test

Verified with the actual Russian Dmitri medium training checkpoint:

- checkpoint size: 845,898,328 bytes
- checkpoint epoch: 5589
- global step: 1,478,840
- sample rate: 22050 Hz
- eSpeak voice: ru

PyTorch 2.6+ compatibility:
- legacy Lightning checkpoints require trusted loading with TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
- PyTorch 2.9+ / 2.11 defaults to the new ONNX exporter, which is not compatible with this old Piper VITS graph
- the notebook patches Piper export_onnx.py to pass dynamo=False
- setup installs onnx and onnxscript

Verified real export:
- ONNX size: 63,516,051 bytes
- onnx.checker: OK
- opset: 15
- inference using the exported ONNX + adjacent .onnx.json: OK
- output WAV: mono, 22050 Hz, 2.067 seconds

Result: REAL_ONNX_INFERENCE_OK

## Qwen3-ASR processor/API test

Verified with Qwen/Qwen3-ASR-1.7B-hf processor without loading the full 1.7B weights:

- processor class: Qwen3ASRProcessor
- apply_transcription_request accepts local file paths or numpy arrays
- expected feature-extractor sample rate: 16000 Hz
- BatchFeature.to(device, dtype) keeps integer token tensors as integers and casts floating audio features to float16
- decode(return_format="transcription_only") returns the complete transcription string

A bug was fixed where the notebook indexed [0] on the decoded string and therefore would have kept only the first character.

The web UI now includes a dedicated single-file Qwen test before full dataset processing.
