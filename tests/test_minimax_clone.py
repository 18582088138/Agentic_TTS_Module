"""
MiniMax 克隆缓存单元测试 —— 假 httpx，零网络、零费用。

复测 / Re-run:
    python -m pytest tests/test_minimax_clone.py -q

覆盖什么（docs/issues/005）:
  · 克隆接口 HTTP 200 但 base_resp 非 0 → 必须报错，**不许写缓存**
  · 缓存里的 voice_id 服务端不存在（t2a 回 2054）→ 作废缓存、重克隆一次、合成成功
"""

from __future__ import annotations

import io
import json

import httpx
import numpy as np
import pytest
import soundfile as sf

from agentic_tts.core.config import Config
from agentic_tts.core.errors import SynthError
from agentic_tts.core.types import Mode
from agentic_tts.engines.base import SynthChunk
from agentic_tts.engines.minimax import MiniMaxEngine


def _wav_hex() -> str:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(2400, dtype="float32"), 24000, format="WAV")
    return buf.getvalue().hex()


class _Resp:
    def __init__(self, data: dict):
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._data


class _FakeMiniMax:
    """按 URL 分派的假服务端；`voices` 是服务端真实存在的音色。"""

    def __init__(self, clone_code: int = 0):
        self.clone_code = clone_code
        self.voices: set[str] = set()
        self.calls: list[str] = []

    def client(self, *args, **kwargs):
        server = self

        class _Client:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def post(self, url, json=None, **kwargs):
                step = url.rsplit("/", 1)[-1]
                server.calls.append(step)
                if step == "upload":
                    return _Resp({"file": {"file_id": 1}, "base_resp": {"status_code": 0}})
                if step == "voice_clone":
                    if server.clone_code == 0:
                        server.voices.add(json["voice_id"])
                    return _Resp({"base_resp": {"status_code": server.clone_code,
                                                "status_msg": "clone rejected"}})
                if json["voice_setting"]["voice_id"] not in server.voices:
                    return _Resp({"base_resp": {"status_code": 2054,
                                                "status_msg": "voice id not exist"}})
                return _Resp({"data": {"audio": _wav_hex()},
                              "extra_info": {"audio_sample_rate": 24000},
                              "base_resp": {"status_code": 0}})

        return _Client()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                 # 缓存路径是相对的 voices/…
    monkeypatch.setenv("TTS_API_KEY", "sk-cp-fake")
    (tmp_path / "ref.mp3").write_bytes(b"fake-ref-audio")
    server = _FakeMiniMax()
    monkeypatch.setattr(httpx, "Client", server.client)
    chunk = SynthChunk(text="你好", mode=Mode.VOICE_CLONE,
                       ref_audio=str(tmp_path / "ref.mp3"), x_vector_only=True)
    return server, chunk, tmp_path / "voices" / "api_voice_cache.json"


def test_failed_clone_raises_and_is_not_cached(setup):
    server, chunk, cache = setup
    server.clone_code = 1004
    engine = MiniMaxEngine(Config())
    with pytest.raises(SynthError, match="克隆音色失败：1004"):
        engine.synthesize(chunk)
    assert not cache.exists() or json.loads(cache.read_text(encoding="utf-8")) == {}
    assert "t2a_v2" not in server.calls


def test_stale_cached_voice_is_recloned_once(setup):
    server, chunk, cache = setup
    engine = MiniMaxEngine(Config())
    engine.synthesize(chunk)                    # 首次：克隆并缓存
    stale = next(iter(engine._cache.values()))["voice_id"]
    server.voices.clear()                       # 服务端把音色回收了
    server.calls.clear()

    audio = engine.synthesize(chunk)

    fresh = audio.effective_gen["voice_id_used"]
    assert server.calls == ["t2a_v2", "upload", "voice_clone", "t2a_v2"]
    assert fresh in server.voices
    saved = [v["voice_id"] for v in json.loads(cache.read_text(encoding="utf-8")).values()]
    assert stale not in saved and fresh in saved
