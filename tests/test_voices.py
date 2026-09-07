"""
音色档案单元测试 / Voice-profile unit tests.

复测 / Re-run:
    python -m pytest tests/test_voices.py -q

重点 / The point:
音色名拼错时**必须报错**，不能静默回落到默认音色 —— 那会得到一条听起来
"好像不太对但也说不清哪里不对"的音频，是最难查的一类问题。
"""

from __future__ import annotations

import pytest

from agentic_tts.core.errors import ConfigError
from agentic_tts.core.types import Mode, SynthRequest, VoiceInfo
from agentic_tts.voices.store import VoiceStore


@pytest.fixture()
def store(config) -> VoiceStore:
    builtin = [
        VoiceInfo(name="Serena", mode=Mode.CUSTOM_VOICE, engine="qwen3", language="Chinese"),
        VoiceInfo(name="Ryan", mode=Mode.CUSTOM_VOICE, engine="qwen3", language="English"),
    ]
    return VoiceStore(config, builtin=builtin)


def test_builtin_lookup_is_case_insensitive(store) -> None:
    assert store.get("serena") is not None


def test_unknown_voice_raises_and_lists_options(store) -> None:
    with pytest.raises(ConfigError, match="可选"):
        store.resolve(SynthRequest(text="x", voice="Serna"))


def test_custom_voice_resolves_speaker_from_name(store) -> None:
    resolved = store.resolve(SynthRequest(text="x", voice="Serena"))
    assert resolved.mode is Mode.CUSTOM_VOICE
    assert resolved.speaker == "Serena"


def test_request_instruct_overrides_profile(store) -> None:
    store.save(VoiceInfo(name="anchor", mode=Mode.CUSTOM_VOICE, engine="qwen3",
                         speaker="Serena", instruct="语速平缓"))
    resolved = store.resolve(SynthRequest(text="x", voice="anchor", instruct="语速偏快"))
    assert resolved.instruct == "语速偏快"       # 请求优先
    assert resolved.speaker == "Serena"          # 其余仍用档案


def test_profile_supplies_clone_parameters(store, tmp_path) -> None:
    ref = store.config.voices_path / "refs" / "a.wav"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(b"fake")
    store.save(VoiceInfo(name="mine", mode=Mode.VOICE_CLONE, engine="qwen3",
                         ref_audio="refs/a.wav", ref_text="参考原话"))
    resolved = store.resolve(SynthRequest(text="x", voice="mine"))
    assert resolved.mode is Mode.VOICE_CLONE
    # 相对路径相对 voices/ 解析 —— 统一相对一个根，避免"搬了一半"
    assert resolved.ref_audio == str(ref)
    assert resolved.ref_text == "参考原话"


def test_icl_profile_without_ref_text_raises(store) -> None:
    store.save(VoiceInfo(name="bad", mode=Mode.VOICE_CLONE, engine="qwen3",
                         ref_audio="refs/a.wav"))
    with pytest.raises(ConfigError, match="ref_text"):
        store.resolve(SynthRequest(text="x", voice="bad"))


def test_x_vector_only_profile_needs_no_ref_text(store) -> None:
    store.save(VoiceInfo(name="xv", mode=Mode.VOICE_CLONE, engine="qwen3",
                         ref_audio="refs/a.wav", x_vector_only=True))
    resolved = store.resolve(SynthRequest(text="x", voice="xv"))
    assert resolved.x_vector_only is True


def test_saved_profile_survives_reload(store) -> None:
    store.save(VoiceInfo(name="calm", mode=Mode.VOICE_DESIGN, engine="qwen3",
                         instruct="沉稳女声"))
    store.reload()
    assert store.get("calm").instruct == "沉稳女声"


def test_delete_profile(store) -> None:
    store.save(VoiceInfo(name="tmp", mode=Mode.VOICE_DESIGN, engine="qwen3", instruct="x"))
    assert store.delete("tmp") is True
    assert store.delete("tmp") is False
    assert store.get("tmp") is None


def test_profile_shadows_builtin_of_the_same_name(store) -> None:
    """用户显式写下的档案应该能覆盖内置默认。"""
    store.save(VoiceInfo(name="Serena", mode=Mode.CUSTOM_VOICE, engine="qwen3",
                         speaker="Serena", instruct="播音腔"))
    assert store.get("Serena").instruct == "播音腔"


def test_role_voice_mapping(store) -> None:
    store.config.voices.roles = {"旁白": "Serena"}
    assert store.role_voice("旁白") == "Serena"
    assert store.role_voice("没配过的角色") is None
