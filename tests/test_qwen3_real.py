"""
Qwen3-TTS 真实模型测试 / Real-model tests —— **默认不跑**（标了 `real`）。

复测 / Re-run（手动，CPU 上每条几十秒到几分钟）:
    conda activate ov_env_py312
    cd Agent_TTS_Module
    python -m pytest tests/test_qwen3_real.py -q -m real -s          # 全部（会载三份权重）
    python -m pytest tests/test_qwen3_real.py -q -m real -s -k custom_voice   # 只跑最便宜的一条

前置 / Prerequisites:
  · `configs/config.yaml` 的 `engine.models_dir` 指向真实权重目录
  · 三个 checkpoint 齐备（`tts doctor` 一眼能看出缺哪个）

为什么只跑短句、只跑一条 / Why one short sentence each:
1.7B 在 CPU 上合成一条 10 字短句是几十秒量级，三个模式各一条已经能覆盖
「权重能载、API 能调、参数能记、轮动加载有效」。音质与参数扫描不属于单测，
那是 verify_qwen3_tts 的活。
"""

from __future__ import annotations

import pytest

from agentic_tts.core.config import Config
from agentic_tts.core.types import Mode, SynthRequest

pytestmark = pytest.mark.real

TEXT = "今天的人工智能资讯。"


@pytest.fixture(scope="module")
def real_module():
    """
    真实门面 / The real facade —— 读项目配置，产物落 `outputs/pytest_real/`。

    `scope="module"` 是刻意的：**权重加载是几十秒**，每个用例重载一次不可接受。
    同一个引擎实例还能顺带验证轮动加载（换模式时先卸再载）。
    """
    from agentic_tts.engine import TTSModule

    config = Config.load()
    if config.engine.backend not in ("qwen3", "qwen3_tts"):
        pytest.skip(f"配置里的引擎是 {config.engine.backend}，本文件只测 qwen3")
    for kind in ("custom_voice", "voice_design", "base"):
        if not config.checkpoint_path(kind).is_dir():
            pytest.skip(f"缺少 checkpoint：{config.checkpoint_path(kind)}")

    module = TTSModule(config)
    yield module
    module.release()


def test_custom_voice(real_module) -> None:
    """内置音色 —— 最便宜的一条，也是唯一必须过的一条。"""
    result = real_module.synthesize(
        SynthRequest(text=TEXT, voice="Serena"), name="pytest_real")
    segment = result.segments[0]
    assert result.ok and result.seconds > 0.5
    assert segment.sample_rate > 0
    # 「实际生效」的参数必须记下来 —— 事后想知道这条音频是怎么来的
    assert segment.effective_gen
    print(f"\ncustom_voice: {result.seconds:.2f}s RTF {result.rtf:.2f} → {result.path}")


def test_voice_design(real_module) -> None:
    """音色设计会**换 checkpoint** —— 顺带验证轮动加载（先卸再载）。"""
    result = real_module.synthesize(
        SynthRequest(text=TEXT, mode=Mode.VOICE_DESIGN,
                     instruct="沉稳的女声，语速平缓，播音腔"), name="pytest_real")
    assert result.ok and result.seconds > 0.5
    assert real_module.engine.resident == ["voice_design"]      # 只驻留一个
    print(f"\nvoice_design: {result.seconds:.2f}s RTF {result.rtf:.2f} → {result.path}")


def test_voice_clone_x_vector(real_module, tmp_path) -> None:
    """
    纯 x-vector 克隆 —— 参考音频用**上一条自己生成的音频**，
    这样测试不依赖仓库里带一份参考 wav。
    """
    source = real_module.synthesize(
        SynthRequest(text="这是一段用来当参考的音频。", voice="Serena"),
        name="pytest_real")
    result = real_module.synthesize(
        SynthRequest(text=TEXT, mode=Mode.VOICE_CLONE, ref_audio=source.path,
                     x_vector_only=True), name="pytest_real")
    assert result.ok and result.seconds > 0.5
    print(f"\nvoice_clone: {result.seconds:.2f}s RTF {result.rtf:.2f} → {result.path}")


def test_instruct_is_recorded_as_applied(real_module) -> None:
    """
    1.7B 上 instruct 有效，结果里必须标 `instruct_applied=True`。
    （0.6B 会被上游静默丢弃，那时这一位必须是 False —— 静默是这个坑的形状。）
    """
    result = real_module.synthesize(
        SynthRequest(text=TEXT, voice="Serena", instruct="语速偏快"), name="pytest_real")
    assert result.segments[0].instruct_applied is True


def test_url_in_text_is_removed_before_synthesis(real_module) -> None:
    """
    实测最值得做的一条预处理：URL 不处理时 CER 4.43（模型逐字母念网址）。
    这里只断言"没送进模型"，音质判断交给耳朵。
    """
    result = real_module.synthesize(
        SynthRequest(text="详情见 https://qwen.ai 谢谢。", voice="Serena"),
        name="pytest_real")
    assert "qwen" not in result.segments[0].spoken_text
