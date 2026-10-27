"""Test scaffolding: heavy deps (torch, the engine) are stubbed — tests validate
the HTTP layer and glue logic of server.py with a fake model, no GPU needed."""

import contextlib
import os
import sys
import tempfile
import types

import numpy as np
import pytest

# Isolated voices dir, fixed BEFORE server.py is imported.
VOICES_DIR = os.path.join(tempfile.mkdtemp(prefix="tts-server-tests-"), "voices")
os.environ["VOICES_DIR"] = VOICES_DIR


def _ensure_torch_stub() -> None:
    """Minimal torch stub: enough for server.py's inference path (no real tensor ops)."""
    try:
        import torch  # noqa: F401  # real torch available

        return
    except Exception:
        pass
    torch = types.ModuleType("torch")
    torch.float16 = np.float16
    torch.float32 = np.float32

    @contextlib.contextmanager
    def inference_mode():
        yield

    seeds = []
    torch.inference_mode = inference_mode
    torch.manual_seed = seeds.append
    torch.test_seeds = seeds  # test hook
    sys.modules["torch"] = torch


def _ensure_lang_map_stub() -> None:
    """Stub omnivoice.utils.lang_map when the engine is not installable in the
    test environment — exercises the same code path the container uses with
    the real 600+ entry map."""
    try:
        import omnivoice.utils.lang_map  # noqa: F401  # real engine available

        return
    except Exception:
        pass
    lang_map = types.ModuleType("omnivoice.utils.lang_map")
    lang_map.LANG_NAME_TO_ID = {
        name: code
        for name, code in zip(
            ["english", "french", "german", "spanish", "italian", "japanese",
             "chinese", "arabic", "russian", "portuguese"],
            ["en", "fr", "de", "es", "it", "ja", "zh", "ar", "ru", "pt"],
        )
    }
    lang_map.LANG_IDS = set(lang_map.LANG_NAME_TO_ID.values())
    sys.modules["omnivoice"] = types.ModuleType("omnivoice")
    sys.modules["omnivoice.utils"] = types.ModuleType("omnivoice.utils")
    sys.modules["omnivoice.utils.lang_map"] = lang_map


_ensure_lang_map_stub()


_ensure_torch_stub()

import server  # noqa: E402  (import after stubbing)


class FakeModel:
    """Stands in for omnivoice.OmniVoice; records generate() kwargs."""

    sampling_rate = 24000

    def __init__(self):
        self.calls = []
        self.fail = None

    def generate(self, text, **kwargs):
        from types import SimpleNamespace

        self.calls.append(SimpleNamespace(text=text, **kwargs))
        if self.fail is not None:
            raise self.fail
        return [np.zeros(240, dtype="float32")]

    def create_voice_clone_prompt(self, wav_path, ref_text=None):
        raise RuntimeError("no prompts in tests")


@pytest.fixture()
def fake_model():
    server._model = FakeModel()
    server._sr = FakeModel.sampling_rate
    yield server._model
    server._model = None


@pytest.fixture()
def client():
    # No context manager: FastAPI startup (real model load) must NOT run.
    from starlette.testclient import TestClient

    return TestClient(server.app)


@pytest.fixture()
def voices_dir_clean():
    for name in os.listdir(VOICES_DIR):
        os.unlink(os.path.join(VOICES_DIR, name))
    yield VOICES_DIR
