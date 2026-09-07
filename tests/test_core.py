"""
配置 / 类型 / 注册表的单元测试 / Unit tests for config, types, and the registry.

复测 / Re-run:
    python -m pytest tests/test_core.py -q

覆盖什么 / What this covers:
  · 配置校验与路径解析（相对路径统一相对项目根、checkpoint 相对 models_dir）
  · 环境变量覆盖
  · **模式推断**：三个 API 各绑一个 checkpoint，推断错就是加载错 4 GB 权重
  · 请求自校验：参数与 mode 冲突时必须报错，不能静默忽略
  · 注册表懒加载：查能力表不该拉起 torch
"""

from __future__ import annotations

import sys

import pytest

from agentic_tts.core.config import PROJECT_ROOT, Config
from agentic_tts.core.errors import ConfigError
from agentic_tts.core.registry import engine_specs, get_engine_class
from agentic_tts.core.types import Capability, Mode, SynthRequest, infer_mode


# ---------------------------------------------------------------- 配置 / config


def test_default_config_is_valid() -> None:
    cfg = Config()
    cfg.validate_runtime()
    assert cfg.engine.backend == "qwen3"
    # 8 GB 卡的前提：默认只驻留一个 checkpoint
    assert cfg.engine.max_resident == 1


def test_repo_dir_defaults_to_vendored_copy() -> None:
    """默认必须指向本仓库内的 vendored 源码，不引用外部项目。"""
    assert Config().repo_path == PROJECT_ROOT / "third_party" / "qwen3_tts"


def test_checkpoint_resolves_against_models_dir(tmp_path) -> None:
    cfg = Config()
    cfg.engine.models_dir = str(tmp_path)
    assert cfg.checkpoint_path("base") == tmp_path / "Qwen3-TTS-12Hz-1.7B-Base"


def test_absolute_checkpoint_is_honoured(tmp_path) -> None:
    cfg = Config()
    cfg.engine.checkpoints["base"] = str(tmp_path / "elsewhere")
    assert cfg.checkpoint_path("base") == tmp_path / "elsewhere"


def test_unknown_checkpoint_kind_raises() -> None:
    with pytest.raises(ConfigError, match="没有 checkpoint"):
        Config().checkpoint_path("nope")


def test_invalid_dtype_raises() -> None:
    cfg = Config()
    cfg.engine.dtype = "float8"
    with pytest.raises(ConfigError, match="dtype"):
        cfg.validate_runtime()


def test_max_chars_must_exceed_min_chars() -> None:
    cfg = Config()
    cfg.text.max_chars = 8
    cfg.text.min_chars = 8
    with pytest.raises(ConfigError, match="max_chars"):
        cfg.validate_runtime()


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv("TTS_BACKEND", "sine")
    monkeypatch.setenv("TTS_MAX_RESIDENT", "2")
    cfg = Config.load()
    assert cfg.engine.backend == "sine"
    assert cfg.engine.max_resident == 2


def test_env_override_rejects_garbage(monkeypatch) -> None:
    monkeypatch.setenv("TTS_MAX_RESIDENT", "many")
    with pytest.raises(ConfigError, match="TTS_MAX_RESIDENT"):
        Config.load()


def test_missing_config_file_raises() -> None:
    with pytest.raises(ConfigError, match="不存在"):
        Config.load("no/such/config.yaml")


# ------------------------------------------------------------ 模式推断 / mode


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"text": "x", "voice": "Serena"}, Mode.CUSTOM_VOICE),
        ({"text": "x"}, Mode.CUSTOM_VOICE),
        ({"text": "x", "instruct": "沉稳女声"}, Mode.VOICE_DESIGN),
        # 给了音色又给指令 = 用这个音色 + 控语气，仍是 custom_voice
        ({"text": "x", "voice": "Serena", "instruct": "语速快"}, Mode.CUSTOM_VOICE),
        ({"text": "x", "ref_audio": "a.wav", "x_vector_only": True}, Mode.VOICE_CLONE),
    ],
)
def test_infer_mode(kwargs, expected) -> None:
    assert infer_mode(SynthRequest(**kwargs)) is expected
    assert SynthRequest(**kwargs).resolved_mode() is expected


def test_clone_without_ref_text_requires_x_vector_only() -> None:
    """ICL 克隆缺 ref_text 要**在加载权重之前**就拦下来（否则白等几十秒）。"""
    with pytest.raises(ValueError, match="ref_text"):
        SynthRequest(text="x", ref_audio="a.wav")


def test_design_without_instruct_raises() -> None:
    with pytest.raises(ValueError, match="instruct"):
        SynthRequest(text="x", mode=Mode.VOICE_DESIGN)


def test_ref_audio_with_wrong_mode_raises() -> None:
    """参数与 mode 冲突必须报错 —— 静默忽略是最难查的一类问题。"""
    with pytest.raises(ValueError, match="ref_audio"):
        SynthRequest(text="x", mode=Mode.CUSTOM_VOICE, ref_audio="a.wav")


# ---------------------------------------------------------- 注册表 / registry


def test_capability_lookup_does_not_import_torch() -> None:
    """
    GUI 开机就要查能力表；这件事**不该**拉起 torch。

    ⚠️ 本用例只在 torch 尚未被导入时有意义（同一进程里别的用例可能已经导入过），
    所以先看当前状态，只在未导入时断言。
    """
    already = "torch" in sys.modules
    specs = engine_specs()
    names = {spec["name"] for spec in specs}
    assert {"qwen3", "sine", "breeze2", "index2"} <= names
    if not already:
        assert "torch" not in sys.modules


def test_qwen3_declares_no_native_vocal_events() -> None:
    """开源权重里没有原生事件 token —— 能力表不能撒谎（tokenizer 全表已核对）。"""
    caps = get_engine_class("qwen3").declared_capabilities()
    assert Capability.VOCAL_EVENTS not in caps
    assert {Capability.CUSTOM_VOICE, Capability.VOICE_DESIGN,
            Capability.VOICE_CLONE, Capability.INSTRUCT} <= caps


def test_placeholder_engine_reports_itself_clearly(config) -> None:
    from agentic_tts.core.errors import CapabilityError
    from agentic_tts.core.registry import create_engine

    # 占位引擎能查能力表（GUI 需要），但实例化时必须给出能照着做的报错
    assert get_engine_class("index2").declared_capabilities()
    with pytest.raises(CapabilityError, match="未接入"):
        create_engine("index2", config)


def test_unknown_engine_lists_options(config) -> None:
    with pytest.raises(ConfigError, match="可选"):
        get_engine_class("whisper-tts")
