"""
OmniVoice TTS — OpenAI-compatible server with zero-shot voice cloning and voice design.

Wraps k2-fsa's `omnivoice` package behind an OpenAI-compatible `/v1/audio/speech`
endpoint, adds a small voice library for zero-shot cloning, and a `/health` probe.

Three modes, one endpoint:
- a stored "voice" is passed → voice cloning (ref_audio, optionally ref_text);
- `instruct` without voice → voice design (describe the desired voice);
- neither → the model picks a voice itself ("auto voice").

Endpoints
---------
GET  /                     short info page
GET  /health               liveness + engine info
GET  /languages            supported language codes (OmniVoice: 600+ languages)
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
import requests
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response

log = logging.getLogger("omnivoice")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

VOICES_DIR = Path(os.environ.get("VOICES_DIR", "/app/voices"))
DEVICE = os.environ.get("TTS_DEVICE", "cuda")  # "cuda" or "cpu"
MODEL_ID = os.environ.get("TTS_MODEL_ID", "k2-fsa/OmniVoice")
DEFAULT_LANG = os.environ.get("TTS_DEFAULT_LANG", "fr")
SAMPLE_RATE = int(os.environ.get("TTS_SAMPLE_RATE", "24000"))
NORMALIZE_TEXT = os.environ.get("TTS_NORMALIZE_TEXT", "1").lower() not in ("", "0", "false", "no")
REF_WINDOW_S = float(os.environ.get("TTS_REF_WINDOW_S", "15"))  # target clip kept as clone reference
REF_MAX_S = float(os.environ.get("TTS_REF_MAX_S", "20"))       # hard cap: longer degrades cloning
STT_URL = os.environ.get("STT_URL", "http://100.64.0.8:9000/v1/audio/transcriptions")
STT_MODEL = os.environ.get("STT_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2")
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "fr")
STT_TIMEOUT = int(os.environ.get("STT_TIMEOUT", "120"))

# Legacy parameters of the previous engine, accepted but ignored (existing
# clients may still send them).
_IGNORED_LEGACY_PARAMS = ("exaggeration", "cfg_weight", "cfg", "temperature")

AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac")

# Minimal honest fallback used only if omnivoice's language map cannot be
# imported (dev environments without the engine). The container always ships
# omnivoice, so /languages serves the real 600+ entry map in production.
_LANG_FALLBACK = {
    "fr": "French", "en": "English", "de": "German", "es": "Spanish",
    "it": "Italian", "ja": "Japanese", "zh": "Chinese", "ar": "Arabic",
    "ru": "Russian", "pt": "Portuguese",
}


def _languages() -> dict[str, str]:
    """OmniVoice language coverage: code -> display name (600+ entries)."""
    try:
        from omnivoice.utils.lang_map import LANG_NAME_TO_ID
    except Exception as exc:  # noqa: BLE001 - dev / stub environments
        log.warning("omnivoice lang_map unavailable (%s); serving fallback list", exc)
        return dict(_LANG_FALLBACK)
    by_id: dict[str, str] = {}
    for name, code in LANG_NAME_TO_ID.items():
        cur = by_id.get(code)
        # Several dialect names can map to distinct codes; for a same code keep
        # the shortest (most canonical) name, e.g. "en" -> "english".
        if cur is None or len(name) < len(cur):
            by_id[code] = name
    return dict(sorted(by_id.items(), key=lambda kv: kv[1]))


LANGUAGES = _languages()

def _voice_name_to_id(name: str) -> str | None:
    """Resolve a human language name or code to an OmniVoice language id."""
    if name in LANGUAGES:
        return name
    if not name:
        return None
    try:
        from omnivoice.utils.lang_map import LANG_NAME_TO_ID
    except Exception:  # noqa: BLE001 - dev / stub environments
        return None
    code = LANG_NAME_TO_ID.get(name.lower())
    return code


def _resolve_lang(value: str | None, voice: str | None) -> str:
    """Requested language -> code; falls back to the stored voice or DEFAULT_LANG."""
    lang = value or _lang_of(voice) or DEFAULT_LANG
    if _voice_name_to_id(lang) is None:
        return DEFAULT_LANG
    return lang


VOICES_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="OmniVoice TTS Server", version="2.0.0")

_model = None
_sr = SAMPLE_RATE
_lock = threading.Lock()


def _load() -> None:
    """Load the OmniVoice engine once at startup (downloads from HF on first run)."""
    global _model, _sr
    from omnivoice import OmniVoice

    device_map = "cpu" if DEVICE == "cpu" else (DEVICE if ":" in DEVICE else "cuda:0")
    dtype = torch.float32 if device_map == "cpu" else torch.float16
    model = OmniVoice.from_pretrained(MODEL_ID, device_map=device_map, dtype=dtype)
    _model = model
    _sr = int(getattr(model, "sampling_rate", SAMPLE_RATE) or SAMPLE_RATE)
    log.info("engine ready: model=%s device=%s sr=%s normalize_text=%s",
             MODEL_ID, DEVICE, _sr, NORMALIZE_TEXT)


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


def _prompt_path(name: str) -> Path:
    """Reusable pre-encoded clone prompt (skips reference re-encoding per request)."""
    return VOICES_DIR / f"{name}.prompt.pt"


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


def _ref_text_of(name: str | None) -> str | None:
    """Transcript stored at upload time, if any."""
    if not name:
        return None
    mp = _meta_path(name)
    if mp.is_file():
        try:
            return json.loads(mp.read_text()).get("ref_text") or None
        except Exception:
            pass
    return None


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


def _select_ref_window(data: np.ndarray, sr: int,
                       target_s: float = REF_WINDOW_S, cap_s: float = REF_MAX_S,
                       ) -> np.ndarray | None:
    """Best-energy window of ~`target_s` seconds from a mono waveform.

    OmniVoice exploits at most ~20 s of reference audio; beyond that quality
    degrades. Speech is uneven, so instead of taking the file head we slide a
    fixed window at 1 s steps and keep the one with the highest mean energy.
    Returns ``None`` when nothing needs trimming.
    """
    wl = int(min(target_s, max(cap_s, 0)) * sr)
    n = int(data.shape[0])
    if wl <= 0 or n <= wl:
        return None
    step = max(1, sr)  # 1 s steps
    best_start, best_energy = 0, -1.0
    for start in range(0, n - wl + 1, step):
        energy = float(np.mean(np.abs(data[start:start + wl]))) if n - start >= wl else -1.0
        if energy > best_energy:
            best_start, best_energy = start, energy
    log.info("clone reference: keeping %s s window starting at %.1f s (of %.1f s)",
             wl / sr, best_start / sr, n / sr)
    return data[best_start:best_start + wl]


def _stt_transcribe(path: Path, language: str) -> str | None:
    """Transcribe the reference clip with the homelab STT service (best-effort).

    Returns the transcript or ``None``; failure degrades cloning quality but is
    never an error for the client.
    """
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                STT_URL,
                files={"file": (path.name, f, "audio/wav")},
                data={"model": STT_MODEL, "language": language},
                timeout=STT_TIMEOUT,
            )
            resp.raise_for_status()
            text = (resp.json().get("text") or "").strip()
    except Exception as exc:  # noqa: BLE001 - degradation, not a hard error
        log.warning("STT failed (%s); voice stored without ref_text (cloning will be degraded)", exc)
        return None
    log.info("STT ok: %d chars of ref_text stored", len(text))
    return text or None


def _precompute_clone_prompt(name: str, wav_path: Path, ref_text: str | None) -> None:
    """Encode the reference once into a reusable prompt file (best-effort)."""
    model = _model
    if model is None:
        return
    try:
        prompt = model.create_voice_clone_prompt(str(wav_path), ref_text=ref_text)
        prompt.save(str(_prompt_path(name)))
        log.info("clone prompt precomputed: %s.prompt.pt", name)
    except Exception as exc:  # noqa: BLE001 - generation falls back to ref_audio
        log.warning("could not precompute clone prompt (%s); cloning will use ref_audio at request time", exc)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (
        "<h1>OmniVoice TTS Server</h1>"
        "<p>OpenAI-compatible TTS with zero-shot voice cloning and voice design "
        "(OmniVoice, 600+ languages).</p>"
        "<ul><li><code>POST /v1/audio/speech</code></li>"
        "<li><code>POST /voices</code></li>"
        "<li><code>GET /v1/voices</code>, <code>GET /languages</code>, <code>GET /health</code></li>"
        "<li><a href='/docs'>/docs</a></li></ul>"
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "engine": "omnivoice",
        "model_id": MODEL_ID,
        "device": DEVICE,
        "sample_rate": SAMPLE_RATE,
        "model_loaded": _model is not None,
        "normalize_text": NORMALIZE_TEXT,
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
    name = _voice_name_to_id(language)  # accepts ids ("fr") or names ("french")
    if name is None:
        raise HTTPException(status_code=400, detail=f"unsupported language '{language}'")
    language = name
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

    # 1-2. 24 kHz mono is done; trim to the best ~15 s window (cap 20 s).
    ref_s = None
    try:
        wav, sr = sf.read(str(dst))
        sliced = _select_ref_window(np.asarray(wav), sr)
        if sliced is not None:
            sf.write(str(dst), sliced, sr)
            ref_s = round(sliced.shape[0] / sr, 2)
    except Exception as exc:  # noqa: BLE001 - oversized references degrade cloning
        log.warning("clone-window trimming failed (%s); keeping the full normalized clip", exc)

    # 3. transcript via the homelab STT — big cloning quality win, best-effort.
    ref_text = _stt_transcribe(dst, language)
    _meta_path(voice_name).write_text(json.dumps({
        "language": language,
        "source": voice_file.filename,
        "ref_text": ref_text,
        "ref_window_s": ref_s,
        "stt_ok": ref_text is not None,
    }))
    _precompute_clone_prompt(voice_name, dst, ref_text)
    log.info("voice stored: %s (lang=%s, ref_text=%s)", dst.name, language, ref_text is not None)
    return {"status": "ok", "voice": voice_name, "language": language, "file": dst.name}


# --------------------------------------------------------------------------- synth
def _encode(wav: object, sr: int, fmt: str) -> tuple[bytes, str]:
    if hasattr(wav, "detach"):  # tolerate a raw torch tensor as well
        wav = wav.detach().cpu().numpy()
    arr = np.asarray(wav, dtype="float32").reshape(-1)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        raw = tf.name
    sf.write(raw, arr, sr)
    out = raw
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


def _coerce(body: dict, key: str, cast, default):
    if key not in body or body[key] is None:
        return default
    try:
        return cast(body[key])
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid '{key}'") from exc


@app.post("/v1/audio/speech")
async def speech(request: Request) -> Response:
    body = await request.json()
    text = (body.get("input") or body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="missing 'input'")

    for legacy in _IGNORED_LEGACY_PARAMS:
        if legacy in body:
            log.warning("parameter '%s' is ignored by the omnivoice engine "
                        "(legacy engine tuning) — kept for API compatibility", legacy)

    num_step = _coerce(body, "num_step", int, 32)
    guidance_scale = _coerce(body, "guidance_scale", float, 2.0)
    speed = _coerce(body, "speed", float, 1.0)
    seed = _coerce(body, "seed", int, None)

    voice = body.get("voice") or body.get("voice_id")
    lang = _resolve_lang(body.get("lang_code") or body.get("language"), voice)
    if lang != (body.get("lang_code") or body.get("language")):
        log.warning("language %r not recognized; falling back to %r", body.get("lang_code"), lang)

    ref = _voice_path(voice)
    if voice and ref is None:
        log.warning("voice %r not found; generating with no reference (auto voice)", voice)

    instruct = (body.get("instruct") or "").strip() or None

    kwargs: dict = {
        "language": lang,
        "num_step": num_step,
        "guidance_scale": guidance_scale,
        "speed": speed,
        "normalize_text": NORMALIZE_TEXT,
    }
    if ref is not None:
        if instruct:
            log.warning("'instruct' ignored while a reference voice is set (mutually exclusive modes)")
            instruct = None
        kwargs["ref_audio"] = str(ref)
        ref_text = _ref_text_of(voice)
        if ref_text:
            kwargs["ref_text"] = ref_text
        # Prefer the pre-encoded prompt (skips reference re-encoding per request).
        pp = _prompt_path(voice)
        if pp.is_file():
            try:
                from omnivoice import VoiceClonePrompt
                kwargs["voice_clone_prompt"] = VoiceClonePrompt.load(str(pp))
                kwargs.pop("ref_audio", None)
                kwargs.pop("ref_text", None)
            except Exception as exc:  # noqa: BLE001 - stale/incompatible prompt file
                log.warning("clone prompt unusable (%s); falling back to ref_audio", exc)
                try:
                    pp.unlink()
                except OSError:
                    pass
    elif instruct is not None:
        kwargs["instruct"] = instruct
    # else: auto voice — neither reference nor instruction.

    with _lock:
        model = _model
        if model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        if seed is not None:
            torch.manual_seed(seed)
        try:
            with torch.inference_mode():
                audio = model.generate(text, **kwargs)
        except ImportError as exc:  # optional text normalizer missing (frontend extra)
            if kwargs.get("normalize_text"):
                log.warning("text normalization unavailable (%s); retrying without normalizing", exc)
                kwargs["normalize_text"] = False
                with torch.inference_mode():
                    audio = model.generate(text, **kwargs)
            else:
                raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc
        except (ValueError, RuntimeError) as exc:
            if isinstance(exc, ValueError) and "instruct" in str(exc).lower():
                raise HTTPException(status_code=400, detail=f"invalid 'instruct': {exc}") from exc
            raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc

    data, mime = _encode(audio[0], _sr, body.get("response_format", "wav"))
    return Response(content=data, media_type=mime)
