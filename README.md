# Chatterbox TTS Server

> 🤖 **This repository is maintained end-to-end by an AI agent**
> ([Nous Research Hermes Agent](https://github.com/NousResearch/hermes-agent)).
> Commits, dependency bumps, base-image updates and image releases are handled
> autonomously — no human operator in the loop.

[![GitHub latest commit](https://badgen.net/github/last-commit/Ace-Nanter/chatterbox-tts-server/main)](https://github.com/Ace-Nanter/chatterbox-tts-server/commits/main/)
[![License:MIT](https://badgen.net/github/license/Ace-Nanter/chatterbox-tts-server)](https://github.com/Ace-Nanter/chatterbox-tts-server/blob/main/LICENSE)
[![Docker](https://img.shields.io/badge/Docker-GPU%20CUDA-blue?logo=docker)](https://www.docker.com/)
[![Chatterbox](https://img.shields.io/badge/Chatterbox-Multilingual%20V3-orange)](https://github.com/resemble-ai/chatterbox)

A self-hosted, **OpenAI-compatible** text-to-speech server wrapping [Resemble AI's Chatterbox](https://github.com/resemble-ai/chatterbox) — **Chatterbox Multilingual V3** (23+ languages) with **zero-shot voice cloning** from a short reference clip. Runs on an NVIDIA GPU (CUDA 12.4).

Drop-in replacement for OpenAI's `/v1/audio/speech`: point any OpenAI TTS client (Open WebUI, LiteLLM, the `openai` Python SDK, …) at it and get fully local, private speech synthesis.

## Features

| Endpoint | Method | Description |
|---|---|---|
| `/v1/audio/speech` | POST | OpenAI-compatible speech synthesis (supports `voice`, `response_format`, `speed`, plus `exaggeration` / `cfg_weight` / `temperature` / `lang_code`) |
| `/voices` | POST | Upload a reference clip to clone a voice (`voice_file`, `voice_name`, `language`) |
| `/v1/voices` | GET | List the stored reference voices |
| `/languages` | GET | List the supported language codes |
| `/health` | GET | Liveness + model info |
| `/docs` | GET | Interactive OpenAPI documentation |

- 🌍 **23+ languages** — Arabic, Chinese, Dutch, **French**, German, Hindi, Italian, Japanese, Korean, Portuguese, Russian, Spanish, … (see `/languages`).
- 🎤 **Zero-shot voice cloning** — clone a speaker from ~10 s of reference audio, no fine-tuning.
- 🗣️ **Emotion exaggeration control** — `exaggeration` (0.25–2.0) and `cfg_weight` (0.0–1.0).
- 🐳 **GPU Docker image** — CUDA 12.4, one container, persistent voice library.

## ⚠️ Why build from a pinned upstream commit (and not PyPI)

The `chatterbox-tts` package on PyPI is **stale**: its latest release (`0.1.7`) predates the
Multilingual **V3** checkpoint, while upstream `master` carries it. Upstream also publishes no
reliable release tags. This image therefore installs Chatterbox from
`github.com/resemble-ai/chatterbox` at a **pinned commit** (`ARG CHATTERBOX_REF`), which is kept
up to date automatically by [`.github/workflows/sync-upstream.yml`](.github/workflows/sync-upstream.yml).

## Quick start

CI builds and publishes the image to the maintainer's container registry
(credentials are supplied through repository secrets and never appear in the repository).
Use whatever reference your own registry serves:

```bash
docker run -d --gpus all \
  --name chatterbox \
  -p 4123:4123 \
  -v ./voices:/app/voices \
  -v ./hf_cache:/app/hf_cache \
  your-registry.example.com/chatterbox-tts-server:latest
```

or with Docker Compose:

```yaml
services:
  chatterbox:
    image: your-registry.example.com/chatterbox-tts-server:latest
    container_name: chatterbox
    restart: unless-stopped
    ports:
      - "4123:4123"
    environment:
      CHATTERBOX_T3_MODEL: v3      # v3 (default) or v2
      CHATTERBOX_DEFAULT_LANG: fr
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

The model (~2 GB) is downloaded from Hugging Face on first start into `HF_HOME` (`/app/hf_cache`).

## Usage

### Synthesize (default voice)

```bash
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Bonjour, ceci est un test.", "response_format": "mp3"}' \
  --output out.mp3
```

### Clone a voice, then use it

```bash
# 1. Upload a reference clip (any format: wav, mp3, m4a, ogg, flac…)
curl -X POST http://localhost:4123/voices \
  -F "voice_file=@my_voice.ogg" \
  -F "voice_name=athena" \
  -F "language=fr"

# 2. Synthesize in that voice
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Bonjour, je suis Athéna.", "voice": "athena"}' \
  --output out.wav
```

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
| `CHATTERBOX_T3_MODEL` | `v3` | Multilingual checkpoint: `v3` or `v2` |
| `CHATTERBOX_DEVICE` | `cuda` | `cuda` or `cpu` |
| `CHATTERBOX_DEFAULT_LANG` | `en` | Default language code when the voice has none |
| `CHATTERBOX_SAMPLE_RATE` | `24000` | Output sample rate |
| `VOICES_DIR` | `/app/voices` | Persistent reference-voice library |
| `HF_HOME` | `/app/hf_cache` | Hugging Face model cache |

## Building

```bash
docker build -t chatterbox-tts-server .
```

## Automation

- **`docker-image.yml`** — builds and pushes on every push to `main` (`latest`) and on `v*` tags; pull requests build without pushing.
- **`sync-upstream.yml`** — weekly check that re-pins the upstream Chatterbox commit, so the image always tracks the official upstream without manual action.
- **`dependabot.yml` + `dependabot-auto-merge.yml`** — keep the base image, GitHub Actions and Python dependencies fresh, auto-merged.

## Credits

- Model: [Resemble AI — Chatterbox](https://github.com/resemble-ai/chatterbox) (MIT).
- Licence: [MIT](LICENSE).

## Donate

<span align="center">

<br />

[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=DX7SKZKNE3E5U)

</span>
