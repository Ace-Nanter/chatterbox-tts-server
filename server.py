"""
Chatterbox TTS — OpenAI-compatible server with zero-shot voice cloning.

Wraps Resemble AI's official `chatterbox-tts` package (Chatterbox Multilingual V3,
23+ languages) behind an OpenAI-compatible `/v1/audio/speech` endpoint, adds a small
voice library for zero-shot cloning, and a `/health` probe.

Voice cloning is zero-shot: a "voice" is just a reference clip stored on disk and
passed to the model at every generation. No training, no fine-tuning at runtime.

Endpoints
---------
GET  /                     short info page
GET  /health               liveness + model info
GET  /languages            supported language codes
GET  /v1/voices            list stored reference voices
POST /voices               multipart upload: voice_file, voice_name, language?
POST /v1/audio/speech      OpenAI-compatible speech synthesis
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response

log = logging.getLogger("chatterbox")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

VOICES_DIR = Path(os.environ.get("VOICES_DIR", "/app/voices"))
DEVICE = os.environ.get("CHATTERBOX_DEVICE", "cuda")
T3_MODEL = os.environ.get("CHATTERBOX_T3_MODEL", "v3")  # "v3" (default) or "v2"
DEFAULT_LANG = os.environ.get("CHATTERBOX_DEFAULT_LANG", "en")
SAMPLE_RATE = int(os.environ.get("CHATTERBOX_SAMPLE_RATE", "24000"))

# Language codes supported by Chatterbox Multilingual.
LANGUAGES = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "ms": "Malay", "nl": "Dutch",
    "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "ru": "Russian",
    "sv": "Swedish", "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}

AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac")

VOICES_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Chatterbox TTS Server", version="1.0.0")

_model = None
_sr = SAMPLE_RATE
_lock = threading.Lock()


def _load() -> None:
    """Load the multilingual model once at startup (downloads from HF on first run)."""
    global _model, _sr
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE, t3_model=T3_MODEL)
    _model = model
    _sr = int(getattr(model, "sr", SAMPLE_RATE))
    log.info("model ready: t3=%s device=%s sr=%s", T3_MODEL, DEVICE, _sr)


@app.on_event("startup")
def _startup() -> None:
    _load()


# --------------------------------------------------------------------------- voices
def _voice_path(name: str | None) -> Path | None:
    if not name:
        return None
    for ext in ("",) + AUDIO_EXTS:
        p = VOICES_DIR / (name if ext == "" else f"{name}{ext}")
        if p.is_file():
            return p
    return None


def _meta_path(name: str) -> Path:
    return VOICES_DIR / f"{name}.json"


def _lang_of(name: str | None) -> str:
    if not name:
        return DEFAULT_LANG
    mp = _meta_path(name)
    if mp.is_file():
        try:
            return json.loads(mp.read_text()).get("language") or DEFAULT_LANG
        except Exception:
            pass
    return DEFAULT_LANG


def _normalize_to_wav(src: Path, dst: Path) -> None:
    """Transcode any input (incl. ogg/m4a) to 24 kHz mono WAV for a clean clone reference."""
    sr = SAMPLE_RATE
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(sr), "-f", "wav", str(dst)],
            check=True, capture_output=True,
        )
    except Exception as exc:  # noqa: BLE001 - fall back to raw bytes if ffmpeg fails
        log.warning("ffmpeg normalize failed (%s); storing raw upload", exc)
        dst.write_bytes(src.read_bytes())


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (
        "<h1>Chatterbox TTS Server</h1>"
        "<p>OpenAI-compatible TTS with zero-shot voice cloning "
        "(Chatterbox Multilingual V3).</p>"
        "<ul><li><code>POST /v1/audio/speech</code></li>"
        "<li><code>POST /voices</code></li>"
        "<li><code>GET /v1/voices</code>, <code>GET /languages</code>, <code>GET /health</code></li>"
        "<li><a href='/docs'>/docs</a></li></ul>"
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "t3_model": T3_MODEL,
        "device": DEVICE,
        "sample_rate": SAMPLE_RATE,
        "model_loaded": _model is not None,
    }


@app.get("/languages")
def languages() -> dict:
    return {"languages": [{"code": c, "name": n} for c, n in LANGUAGES.items()]}


@app.get("/v1/voices")
def list_voices() -> dict:
    out = []
    for p in sorted(VOICES_DIR.glob("*")):
        if p.suffix.lower() in AUDIO_EXTS:
            out.append({"id": p.stem, "name": p.stem, "language": _lang_of(p.stem), "file": p.name})
    return {"voices": out}


@app.post("/voices")
async def upload_voice(
    voice_file: UploadFile = File(...),
    voice_name: str = Form(...),
    language: str = Form(DEFAULT_LANG),
) -> dict:
    if language not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"unsupported language '{language}'")
    if not all(c.isalnum() or c in "-_." for c in voice_name):
        raise HTTPException(status_code=400, detail="voice_name may only contain [a-zA-Z0-9-_.]")

    raw_ext = Path(voice_file.filename or "ref.wav").suffix.lower() or ".wav"
    tmp = VOICES_DIR / f".upload_{voice_name}{raw_ext}"
    tmp.write_bytes(await voice_file.read())
    dst = VOICES_DIR / f"{voice_name}.wav"
    _normalize_to_wav(tmp, dst)
    try:
        tmp.unlink()
    except OSError:
        pass
    _meta_path(voice_name).write_text(json.dumps({"language": language, "source": voice_file.filename}))
    log.info("voice stored: %s (lang=%s)", dst.name, language)
    return {"status": "ok", "voice": voice_name, "language": language, "file": dst.name}


# --------------------------------------------------------------------------- synth
def _encode(wav: torch.Tensor, sr: int, fmt: str) -> tuple[bytes, str]:
    arr = np.asarray(wav.squeeze().detach().cpu().numpy(), dtype="float32")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        raw = tf.name
    sf.write(raw, arr, sr)
    fmt = (fmt or "wav").lower()
    if fmt in ("wav", "wave"):
        return Path(raw).read_bytes(), "audio/wav"
    codec = {"mp3": "libmp3lame", "opus": "libopus", "flac": "flac", "aac": "aac"}.get(fmt)
    if not codec:
        return Path(raw).read_bytes(), "audio/wav"
    out = raw.rsplit(".", 1)[0] + "." + fmt
    try:
        subprocess.run(["ffmpeg", "-y", "-i", raw, "-c:a", codec, out], check=True, capture_output=True)
        mime = {"mp3": "audio/mpeg", "opus": "audio/ogg", "flac": "audio/flac", "aac": "audio/aac"}[fmt]
        return Path(out).read_bytes(), mime
    except Exception as exc:  # noqa: BLE001
        log.warning("ffmpeg encode to %s failed (%s); returning wav", fmt, exc)
        return Path(raw).read_bytes(), "audio/wav"
    finally:
        for f in (raw, out):
            try:
                os.unlink(f)
            except OSError:
                pass


@app.post("/v1/audio/speech")
async def speech(request: Request) -> Response:
    body = await request.json()
    text = (body.get("input") or body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="missing 'input'")

    voice = body.get("voice") or body.get("voice_id")
    lang = body.get("lang_code") or body.get("language") or _lang_of(voice)
    if lang not in LANGUAGES:
        lang = DEFAULT_LANG

    ref = _voice_path(voice)
    if voice and ref is None:
        log.warning("voice %r not found; using the model default voice", voice)

    kwargs = {
        "exaggeration": float(body.get("exaggeration", 0.5)),
        "cfg_weight": float(body.get("cfg_weight", body.get("cfg", 0.5))),
        "temperature": float(body.get("temperature", 0.8)),
    }
    if ref is not None:
        kwargs["audio_prompt_path"] = str(ref)

    with _lock:
        model = _model
        if model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        with torch.inference_mode():
            wav = model.generate(text, language_id=lang, **kwargs)

    data, mime = _encode(wav, _sr, body.get("response_format", "wav"))
    return Response(content=data, media_type=mime)
