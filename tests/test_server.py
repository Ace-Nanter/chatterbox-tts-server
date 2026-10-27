"""Unit tests for tts-server (engine mocked — no GPU, no model download)."""

import io
import json
import os

import numpy as np
import pytest

import server
from server import _select_ref_window, _lang_of, _ref_text_of

AUDIO_BYTES = b"\x00" * 64  # not decoded by the fake model anyway


# --------------------------------------------------------------------- /health
def test_health_contract(client, fake_model):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    for key in ("status", "device", "sample_rate", "model_loaded"):
        assert key in body           # unchanged fields
    assert body["engine"] == "omnivoice"
    assert body["model_loaded"] is True


def test_health_without_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is False


# ------------------------------------------------------------------ /languages
def test_languages_shape(client):
    r = client.get("/languages")
    assert r.status_code == 200
    langs = r.json()["languages"]
    assert all(set(e) == {"code", "name"} for e in langs)
    codes = {e["code"] for e in langs}
    assert "fr" in codes and "en" in codes      # fallback map at minimum


# ------------------------------------------------------------- /v1/audio/speech
def _speech(client, **payload):
    payload.setdefault("input", "Bonjour, ceci est un test.")
    return client.post("/v1/audio/speech", json=payload)


def test_speech_clone_voice(client, fake_model, voices_dir_clean):
    open(os.path.join(server.VOICES_DIR, "athena.wav"), "wb").write(AUDIO_BYTES)
    json.dump({"language": "fr", "ref_text": "Transcript de référence."},
              open(os.path.join(server.VOICES_DIR, "athena.json"), "w"))
    r = _speech(client, voice="athena", lang_code="fr")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    call = fake_model.calls[-1]
    assert call.text == "Bonjour, ceci est un test."
    assert call.ref_audio == os.path.join(server.VOICES_DIR, "athena.wav")
    assert call.ref_text == "Transcript de référence."
    assert call.language == "fr"
    assert call.num_step == 32                # new params' defaults
    assert call.guidance_scale == 2.0
    assert call.normalize_text is True
    assert not hasattr(call, "exaggeration")  # legacy params never forwarded


def test_speech_legacy_params_accepted_no_4xx(client, fake_model):
    r = _speech(client, exaggeration=0.7, cfg_weight=0.3, cfg=0.3, temperature=0.9)
    assert r.status_code == 200
    call = fake_model.calls[-1]
    assert not any(hasattr(call, p) for p in ("exaggeration", "cfg_weight", "cfg", "temperature"))


def test_speech_auto_voice(client, fake_model):
    r = _speech(client)
    assert r.status_code == 200
    call = fake_model.calls[-1]
    assert not hasattr(call, "ref_audio")
    assert not hasattr(call, "instruct")


def test_speech_voice_design(client, fake_model):
    r = _speech(client, instruct="female, british accent")
    assert r.status_code == 200
    call = fake_model.calls[-1]
    assert call.instruct == "female, british accent"
    assert not hasattr(call, "ref_audio")


def test_speech_instruct_trumps_on_clone(client, fake_model, voices_dir_clean):
    open(os.path.join(server.VOICES_DIR, "athena.wav"), "wb").write(AUDIO_BYTES)
    r = _speech(client, voice="athena", instruct="female, british accent")
    assert r.status_code == 200
    call = fake_model.calls[-1]
    assert not hasattr(call, "instruct")      # mutually exclusive modes
    assert call.ref_audio


def test_speech_speed_and_seed(client, fake_model):
    r = _speech(client, speed=0.8, seed=42)
    assert r.status_code == 200
    call = fake_model.calls[-1]
    assert call.speed == 0.8
    import torch

    seeds = getattr(torch, "test_seeds", None)   # torch stub hook (absent with real torch)
    if seeds is not None:
        assert seeds[-1] == 42                   # engine seeded deterministically


def test_speech_unknown_lang_falls_back(client, fake_model):
    r = _speech(client, lang_code="xx")
    assert r.status_code == 200
    assert fake_model.calls[-1].language == server.DEFAULT_LANG


def test_speech_custom_num_step(client, fake_model):
    r = _speech(client, num_step=16, guidance_scale=1.5)
    assert r.status_code == 200
    call = fake_model.calls[-1]
    assert call.num_step == 16 and call.guidance_scale == 1.5


def test_speech_invalid_num_step_is_400(client, fake_model):
    assert _speech(client, num_step="fast").status_code == 400


def test_speech_missing_input_is_400(client, fake_model):
    assert _speech(client, input="   ").status_code == 400


def test_speech_unknown_voice_is_auto_voice(client, fake_model):
    r = _speech(client, voice="ghost")
    assert r.status_code == 200
    assert not hasattr(fake_model.calls[-1], "ref_audio")


def test_speech_generation_failure_is_500(client, fake_model):
    fake_model.fail = RuntimeError("GPU sad")
    r = _speech(client)
    assert r.status_code == 500


def test_speech_response_format_mp3(client, fake_model):
    """Unknown/unsupported encodings fall back to wav exactly as before."""
    r = _speech(client, response_format="mp3")
    # no ffmpeg in the test env would still be 200 (wav fallback); assert 200 either way
    assert r.status_code == 200


# -------------------------------------------------------------- voice pipeline
def _wav_bytes(dur_s, sr=24000, amplitude=0.1, offset_s=0.0):
    t = np.linspace(0, dur_s, int(dur_s * sr), endpoint=False)
    tone = (amplitude * np.sin(2 * np.pi * 440 * t)).astype("float32")
    if offset_s:
        tone[: int(offset_s * sr)] = 0.0    # silent head
    buf = io.BytesIO()
    import soundfile as sf

    sf.write(buf, tone, sr, format="WAV")
    buf.seek(0)
    return buf.getvalue()


def test_select_ref_window():
    sr = 24000
    loud = np.full(sr * 40, 0.001, dtype="float32")
    loud[int(30 * sr):] = 0.9                # speech burst at 30 s
    out = _select_ref_window(loud, sr)
    assert out is not None and out.shape[0] == int(15 * sr)
    assert out[-1] > 0.5                     # the window captured the burst


def test_select_ref_window_short_clip_unchanged():
    sr = 24000
    data = np.zeros(sr * 5, dtype="float32")
    assert _select_ref_window(data, sr) is None


def test_upload_transcribes_and_trims(client, fake_model, voices_dir_clean, monkeypatch):
    recorded = {}

    class FakeResp:
        def raise_for_status(self): ...

        def json(self):
            return {"text": "Bonjour, j'écoute et je réponds."}

    def fake_post(url, files=None, data=None, timeout=None):
        recorded["url"], recorded["data"] = url, data
        recorded["file"] = files["file"][1].read()
        return FakeResp()

    import requests

    monkeypatch.setattr(requests, "post", fake_post)
    r = client.post(
        "/voices",
        files={"voice_file": ("ref.wav", io.BytesIO(_wav_bytes(40, amplitude=0.5)), "audio/wav")},
        data={"voice_name": "athena", "language": "fr"},
    )
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "voice": "athena", "language": "fr", "file": "athena.wav"}
    # metadata carries the STT transcript
    meta = json.load(open(os.path.join(server.VOICES_DIR, "athena.json")))
    assert meta["ref_text"] == "Bonjour, j'écoute et je réponds."
    assert meta["stt_ok"] is True and meta["language"] == "fr"
    # reference was trimmed to <= 20 s (40 s input)
    import soundfile as sf

    wav, sr = sf.read(os.path.join(server.VOICES_DIR, "athena.wav"))
    assert wav.shape[0] <= 20 * sr
    assert recorded["url"] == server.STT_URL
    assert recorded["data"]["model"] == server.STT_MODEL
    assert recorded["data"]["language"] == "fr"


def test_upload_stt_failure_degrades_silently(client, fake_model, voices_dir_clean, monkeypatch):
    import requests

    def boom(*a, **kw):
        raise OSError("STT down")

    monkeypatch.setattr(requests, "post", boom)
    r = client.post(
        "/voices",
        files={"voice_file": ("ref.wav", io.BytesIO(_wav_bytes(3)), "audio/wav")},
        data={"voice_name": "fallback", "language": "fr"},
    )
    assert r.status_code == 200  # degradation, not an error
    meta = json.load(open(os.path.join(server.VOICES_DIR, "fallback.json")))
    assert meta["ref_text"] is None and meta["stt_ok"] is False
    assert _ref_text_of("fallback") is None


def test_upload_rejects_unsupported_language(client, fake_model, voices_dir_clean):
    r = client.post(
        "/voices",
        files={"voice_file": ("ref.wav", io.BytesIO(_wav_bytes(1)), "audio/wav")},
        data={"voice_name": "bad", "language": "not-a-language"},
    )
    assert r.status_code == 400


def test_upload_rejects_bad_name(client, fake_model, voices_dir_clean):
    r = client.post(
        "/voices",
        files={"voice_file": ("ref.wav", io.BytesIO(_wav_bytes(1)), "audio/wav")},
        data={"voice_name": "bad name!", "language": "fr"},
    )
    assert r.status_code == 400


def test_voice_language_names_accepted(client, fake_model, voices_dir_clean):
    r = client.post(
        "/voices",
        files={"voice_file": ("ref.wav", io.BytesIO(_wav_bytes(1)), "audio/wav")},
        data={"voice_name": "ansi", "language": "english"},
    )
    assert r.status_code == 200
    assert r.json()["language"] == "en"
    assert _lang_of("ansi") == "en"


def test_list_voices(client, fake_model, voices_dir_clean):
    open(os.path.join(server.VOICES_DIR, "athena.wav"), "wb").write(AUDIO_BYTES)
    r = client.get("/v1/voices")
    assert r.status_code == 200
    entries = r.json()["voices"]
    assert len(entries) == 1 and entries[0]["id"] == "athena"
