"""
VoiceClone 端到端测试 / End-to-end test for voice_clone flow.

不依赖真 MiniMax API，用 mock HTTP server 验证：
1. APITTSEngine.synthesize() mode=voice_clone 时
2. 自动 upload ref_audio → 拿 voice_id
3. 缓存命中：第二次不走 upload
4. synthesize 返回 RawAudio

简化：mock MiniMax /v1/files/upload + /v1/voice_clone + /v1/t2a_v2。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

# 让脚本能直接跑
sys.path.insert(0, "/workspace/Agentic_TTS_Module")


@pytest.fixture
def mock_minimax(monkeypatch):
    """mock httpx.post 让 MiniMax 不发真请求。"""
    import httpx
    import io
    import numpy as np
    import soundfile as sf

    # 真实的静音 wav（soundfile 能解码）
    buf = io.BytesIO()
    sf.write(buf, np.zeros(12000, dtype="float32"), 24000, format="WAV")
    fake_wav = buf.getvalue()

    uploaded = {"calls": 0, "files": []}
    cloned = {"calls": 0, "voice_ids": []}
    synthesized = {"calls": 0}

    def fake_post(url, headers=None, json=None, files=None, data=None, timeout=None):
        # 文件上传
        if "/files/upload" in url:
            uploaded["calls"] += 1
            fid = 99000 + uploaded["calls"]
            uploaded["files"].append({"file_id": fid, "data_url": url})
            return httpx.Response(200, json={"file": {"file_id": fid}})

        # voice_clone
        if "/voice_clone" in url:
            cloned["calls"] += 1
            vid = json.get("voice_id", "voice_unknown")
            cloned["voice_ids"].append(vid)
            return httpx.Response(200, json={
                "voice_id": vid,
                "trial_audio": "",
                "base_resp": {"status_code": 0, "status_msg": "success"},
            })

        # t2a_v2
        if "/t2a_v2" in url:
            synthesized["calls"] += 1
            return httpx.Response(200, json={
                "audio_file": fake_wav.hex(),
                "trace_id": "trace_001",
                "extra_info": {"audio_length": 0.5, "audio_sample_rate": 24000},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            })

        return httpx.Response(404, json={"error": "not found", "path": url})

    monkeypatch.setattr(httpx, "post", fake_post)
    return {"uploaded": uploaded, "cloned": cloned, "synthesized": synthesized}


def test_voice_clone_full_flow(tmp_path, mock_minimax, monkeypatch):
    """端到端：上传 → clone → 合成，第二轮 cache hit。"""
    monkeypatch.setenv("MINIMAX_API_KEY", "mock-key")
    # 用临时 voice_cache 路径
    cache_path = tmp_path / "api_voice_cache.json"

    from agentic_tts.core.config import Config, APIConfig
    from agentic_tts.engines.api.engine import APITTSEngine
    from agentic_tts.engines.base import SynthChunk
    from agentic_tts.core.types import Mode

    cfg = Config()
    cfg.api = APIConfig(enabled=True, provider="minimax",
                        api_key="mock-key", voice_cache_path=str(cache_path))

    engine = APITTSEngine(cfg)

    # 准备 ref_audio（用已上传的文件）
    ref_path = Path("/workspace/Agentic_TTS_Module/voices/refs/ref_audio_zh.mp3")
    assert ref_path.exists(), "ref audio must be in repo"

    chunk = SynthChunk(
        text="今天的人工智能资讯",
        mode=Mode.VOICE_CLONE,
        language="Chinese",
        ref_audio=str(ref_path),
        ref_text="今天的人工智能资讯",
    )

    # 第一次合成：触发 upload + clone
    raw1 = engine.synthesize(chunk)
    assert mock_minimax["uploaded"]["calls"] == 1
    assert mock_minimax["cloned"]["calls"] == 1
    assert mock_minimax["synthesized"]["calls"] == 1
    assert raw1.sample_rate == 24000
    voice_id_1 = raw1.effective_gen["voice_id_used"]
    assert voice_id_1, "voice_id 应该被记进 effective_gen"

    # 第二次合成：cache hit，不再 upload
    raw2 = engine.synthesize(chunk)
    assert mock_minimax["uploaded"]["calls"] == 1   # 不变
    assert mock_minimax["cloned"]["calls"] == 1     # 不变
    assert mock_minimax["synthesized"]["calls"] == 2  # 又合了一次
    voice_id_2 = raw2.effective_gen["voice_id_used"]
    assert voice_id_2 == voice_id_1, "cache 命中应该返回同一个 voice_id"

    # 缓存文件存在
    assert cache_path.exists()
    cache_data = json.loads(cache_path.read_text())
    assert len(cache_data) == 1


def test_voice_clone_with_native_event(mock_minimax, monkeypatch):
    """[laugh] 在 voice_clone + API 引擎下应被翻译成 (laughs)。"""
    monkeypatch.setenv("MINIMAX_API_KEY", "mock-key")

    captured_text = {"value": None}

    import httpx
    import io
    import numpy as np
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.zeros(12000, dtype="float32"), 24000, format="WAV")
    fake_wav = buf.getvalue()

    def fake_post_capture(url, headers=None, json=None, **kw):
        if "/t2a_v2" in url and json:
            captured_text["value"] = json.get("text", "")
            return httpx.Response(200, json={
                "audio_file": fake_wav.hex(),
                "extra_info": {"audio_length": 0.5, "audio_sample_rate": 24000},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            })
        return httpx.Response(200, json={"file": {"file_id": 1},
                                          "voice_id": "v1", "base_resp": {"status_code": 0}})
    monkeypatch.setattr(httpx, "post", fake_post_capture)

    from agentic_tts.core.config import Config, APIConfig
    from agentic_tts.engines.api.engine import APITTSEngine
    from agentic_tts.engines.base import SynthChunk
    from agentic_tts.core.types import Mode

    cfg = Config()
    cfg.api = APIConfig(enabled=True, provider="minimax", api_key="mock-key",
                        voice_cache_path="/tmp/test_cache_e2e.json")
    engine = APITTSEngine(cfg)
    ref = Path("/workspace/Agentic_TTS_Module/voices/refs/ref_audio_zh.mp3")
    chunk = SynthChunk(text="我笑了 [laugh] 然后继续",
                       mode=Mode.VOICE_CLONE, language="Chinese",
                       ref_audio=str(ref), ref_text="原话")
    engine.synthesize(chunk)

    assert captured_text["value"] is not None
    assert "(laughs)" in captured_text["value"], f"未翻译：{captured_text['value']}"
    assert "[laugh]" not in captured_text["value"], "标记应被替换"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])