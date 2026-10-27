# TTS Server

> 🤖 **This repository is maintained end-to-end by an AI agent**
> ([Nous Research Hermes Agent](https://github.com/NousResearch/hermes-agent)).
> Commits, dependency bumps, base-image updates and image releases are handled
> autonomously — no human operator in the loop.

[![GitHub latest commit](https://badgen.net/github/last-commit/Ace-Nanter/tts-server/main)](https://github.com/Ace-Nanter/tts-server/commits/main/)
[![License:MIT](https://badgen.net/github/license/Ace-Nanter/tts-server)](https://github.com/Ace-Nanter/tts-server/blob/main/LICENSE)
[![Docker](https://img.shields.io/badge/Docker-GPU%20CUDA-blue?logo=docker)](https://www.docker.com/)
[![Engine](https://img.shields.io/badge/engine-OmniVoice-600%2B%20languages-orange)](https://github.com/k2-fsa/OmniVoice)

A self-hosted, **OpenAI-compatible** text-to-speech server wrapping [k2-fsa's OmniVoice](https://github.com/k2-fsa/OmniVoice) — **600+ languages** with **zero-shot voice cloning** and **voice design** (`instruct`). Runs on an NVIDIA GPU (CUDA 12.8 / torch 2.8.0+cu128).

Drop-in replacement for OpenAI's `/v1/audio/speech`: point any OpenAI TTS client (Open WebUI, LiteLLM, the `openai` Python SDK, …) at it and get fully local, private speech synthesis. Output is **24 kHz mono**.

## Features

| Endpoint | Method | Description |
|---|---|---|
| `/v1/audio/speech` | POST | OpenAI-compatible speech synthesis (`voice`, `response_format`, `speed`, `lang_code`/`language`, `instruct`, `seed`, `num_step`, `guidance_scale`) |
| `/voices` | POST | Upload a reference clip to clone a voice (`voice_file`, `voice_name`, `language`) |
| `/v1/voices` | GET | List the stored reference voices |
| `/languages` | GET | OmniVoice language coverage (600+ codes/names) |
| `/health` | GET | Liveness + engine info (`engine: omnivoice`) |
| `/docs` | GET | Interactive OpenAPI documentation |

- 🌍 **600+ languages** — the full upstream `language_id` map is served by `/languages`.
- 🎤 **Zero-shot voice cloning** — from a short reference clip; best with `~15 s` of clean speech (hard-capped at 20 s — beyond that, cloning quality degrades).
- 📝 **Reference transcripts** — on upload, the clip is transcribed by the homelab STT service and the transcript (`ref_text`) is stored in the voice metadata, which noticeably improves cloning fidelity.
- 🗣️ **Voice design** — describe the desired voice with `instruct` (e.g. `"female, british accent"`); no reference needed. With no voice and no `instruct`, the model picks a voice itself (auto voice).
- 🔢 **Text normalization** — numbers, dates and times (« 18 h 30 ») are converted to their spoken form (`normalize_text`).
- 🐳 **GPU Docker image** — CUDA 12.8, one container, persistent voice library.

⚠️ **License**: the model weights are CC-BY-NC (**non-commercial**) and the audio tokenizer is under a community license — see [`NOTICE`](NOTICE) before any deployment roll-out.

## ⚠️ Why build from a pinned upstream commit (and not PyPI)

Upstream OmniVoice publishes **no usable release tags**, so the image installs the package from `github.com/k2-fsa/OmniVoice` at a **pinned commit** (`ARG OMNIVOICE_REF`), kept up to date automatically by [`.github/workflows/sync-upstream.yml`](.github/workflows/sync-upstream.yml). The engine requires **torch 2.8.0 with the CUDA 12.8 wheels** — the base image and torch pin in the `Dockerfile` must move together.

## Quick start

CI builds and publishes the image to the maintainer's container registry
(credentials are supplied through repository secrets and never appear in the repository).
Use whatever reference your own registry serves:

```bash
docker run -d --gpus all \
  --name tts-server \
  -p 4123:4123 \
  -v ./voices:/app/voices \
  -v ./hf_cache:/app/hf_cache \
  your-registry.example.com/tts-server:latest
```

or with Docker Compose:

```yaml
services:
  tts-server:
    image: your-registry.example.com/tts-server:latest
    container_name: tts-server
    restart: unless-stopped
    ports:
      - "4123:4123"
    environment:
      TTS_DEVICE: cuda
      TTS_DEFAULT_LANG: fr
      TTS_NORMALIZE_TEXT: "1"   # default: speak numbers/dates/times correctly
    volumes:
      - ./voices:/app/voices
      - ./hf_cache:/app/hf_cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

The weights (several GB) are downloaded from Hugging Face on first start into `HF_HOME` (`/app/hf_cache`).

## Usage

### Synthesize (auto voice)

```bash
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Bonjour, ceci est un test.", "lang_code": "fr", "response_format": "mp3"}' \
  --output out.mp3
```

### Clone a voice, then use it

```bash
# 1. Upload a reference clip (any format: wav, mp3, m4a, ogg, flac…).
#    Uploaded audio is normalized to 24 kHz mono, trimmed to the best ~15 s
#    window, transcribed via STT (recommended: clean single-speaker speech).
curl -X POST http://localhost:4123/voices \
  -F "voice_file=@my_voice.ogg" \
  -F "voice_name=athena" \
  -F "language=fr"

# 2. Synthesize in that voice (OpenAI compatible)
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Bonjour, je suis Athéna.", "voice": "athena", "lang_code": "fr"}' \
  --output out.wav
```

### Voice design (no reference clip)

```bash
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Bienvenue au journal.", "lang_code": "fr",
       "instruct": "female, middle-aged"}'
```

### Engine knobs

| Parameter | Default | Description |
|---|---|---|
| `num_step` | `32` | Diffusion decoding steps (`16` is a faster, slightly looser alternative) |
| `guidance_scale` | `2.0` | Classifier-free guidance strength |
| `seed` | — | Deterministic sampling (seeded before each generation) |
| `speed` | `1.0` | Speaking-rate factor |

Legacy parameters (`exaggeration`, `cfg_weight`, `temperature`) are **accepted but ignored** — existing clients never get a 4xx; a warning is logged instead.

### OpenAI-compatible client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:4123/v1", api_key="not-needed")
with client.audio.speech.with_streaming_response.create(
    model="tts-1", voice="athena", input="Bonjour !"
) as resp:
    resp.stream_to_file("out.mp3")
```

## Configuration

| Env variable | Default | Description |
|---|---|---|
| `TTS_DEVICE` | `cuda` | `cuda` or `cpu` |
| `TTS_MODEL_ID` | `k2-fsa/OmniVoice` | Hugging Face repo id (or local path) |
| `TTS_DEFAULT_LANG` | `fr` | Fallback language when none is requested and the voice has none |
| `TTS_SAMPLE_RATE` | `24000` | Output sample rate (the tokenizer's native rate) |
| `TTS_NORMALIZE_TEXT` | `1` | Spoken-form conversion of numbers/dates/times (`0` disables) |
| `TTS_REF_WINDOW_S` | `15` | Target duration of the kept clone-reference window |
| `TTS_REF_MAX_S` | `20` | Hard cap on the clone-reference duration |
| `STT_URL` | `http://100.64.0.8:9000/v1/audio/transcriptions` | Homelab STT (whisper) endpoint used to transcribe uploaded clips |
| `STT_MODEL` | `deepdml/faster-whisper-large-v3-turbo-ct2` | STT model field sent to the endpoint |
| `STT_LANGUAGE` | `fr` | STT language when the clip has none |
| `STT_TIMEOUT` | `120` | STT request timeout (s); on failure the voice is stored without transcript (degraded cloning, no error) |
| `VOICES_DIR` | `/app/voices` | Persistent reference-voice library |
| `HF_HOME` | `/app/hf_cache` | Hugging Face model cache |

## Building

```bash
docker build -t tts-server .
```

## Smoke test (post-deployment)

```bash
./scripts/smoke.sh http://localhost:4123   # default base URL
```

Checks `/health` (engine), `/languages`, then synthesizes a French sentence with the `athena` voice in **wav / mp3 / opus** and validates each payload is non-empty audio of the right MIME type.

## Automation

- **`docker-image.yml`** — unit tests (engine mocked, no GPU) + image build/push on every push to `main` (`latest`) and on `v*` tags; pull requests build without pushing. The image is published as `…/tts-server`; the previous image name is left untouched in the registry as a rollback path.
- **`sync-upstream.yml`** — weekly check that re-pins the upstream OmniVoice commit, so the image always tracks official upstream without manual action.
- **`dependabot.yml` + `dependabot-auto-merge.yml`** — keep the base image, GitHub Actions and Python dependencies fresh, auto-merged (never semver-major bumps).

## Credits

- Model: [k2-fsa — OmniVoice](https://github.com/k2-fsa/OmniVoice) (code Apache-2.0; **weights CC-BY-NC**, tokenizer under a community license — see [`NOTICE`](NOTICE)).
- Licence du code de ce dépôt : [MIT](LICENSE).

## Donate

<span align="center">

<br />

[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=DX7SKZKNE3E5U)

</span>
