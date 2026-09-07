"""
门面端到端测试 / Facade end-to-end tests —— 走 `sine` 引擎，**不加载任何权重**。

复测 / Re-run:
    python -m pytest tests/test_module.py -q

这一层要证明的事 / What must hold here:
  · 文本链路顺序正确（事件标记先解析，再预处理，再分段）
  · 停顿真的变成了静音（时长可断言）
  · 失控能靠换种子救回；救不回的段不混进合并音频
  · 每段单独落盘 + 合并落盘并存；manifest 写全
  · 段种子固定为 base+序号 —— **单段重生成不影响其他段**
  · 多段混模式时按 checkpoint 分组执行（减少 8 GB 卡上的权重轮动）
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_tts.core.errors import CapabilityError, SynthError
from agentic_tts.core.types import BatchSynthRequest, Capability, Mode, SynthRequest


def test_single_segment_writes_one_file(module) -> None:
    result = module.synthesize("这是一句用来测试的话。", voice="test_female", name="run")
    assert result.ok
    assert result.path and Path(result.path).is_file()
    assert result.segments[0].sample_rate == 24000
    assert result.seconds > 0


def test_repeated_auto_named_runs_do_not_overwrite_each_other(module) -> None:
    """
    回归：连续两次生成必须落在**不同**目录。

    时间戳只有秒级精度、段文件名又只由「段序号 + 音色」决定 —— 同一秒内的第二次
    生成曾经静默覆盖第一次，表现就是「点了生成却没有新文件」（见 issues/004）。
    """
    paths = [module.synthesize(f"这是第{i}次点击生成的内容。", voice="test_female").path
             for i in range(3)]
    assert len(set(paths)) == 3
    assert all(Path(p).is_file() for p in paths)


def test_explicit_name_still_reuses_the_directory(module) -> None:
    """显式 `name` 是「写到这个目录」的明确要求，覆盖是调用方自己的选择。"""
    first = module.synthesize("第一次。", voice="test_female", name="fixed")
    second = module.synthesize("第二次。", voice="test_female", name="fixed")
    assert Path(first.path).parent == Path(second.path).parent


def test_pause_markup_becomes_real_silence(module) -> None:
    """
    `[pause:1s]` 必须变成 1 秒静音 —— 这是唯一确定性的节奏控制手段，
    时长可断言正是它"确定"的意思。
    """
    plain = module.synthesize("前半句话在这里。后半句话在这里。", voice="test_male",
                              name="plain", save=False)
    paused = module.synthesize("前半句话在这里。[pause:1000ms]后半句话在这里。",
                               voice="test_male", name="paused", save=False)
    assert paused.seconds == pytest.approx(plain.seconds + 1.0, abs=0.05)


def test_events_warn_when_engine_has_no_native_support(module) -> None:
    result = module.synthesize("这句话很有意思。[laugh]", voice="test_female", save=False)
    assert any("不保证" in w for w in result.segments[0].warnings)


def test_long_text_is_chunked_internally_but_yields_one_file(module) -> None:
    module.config.text.max_chars = 20
    text = "".join(f"这是第{i}句话，内容稍微长一点。" for i in range(1, 6))
    result = module.synthesize(text, voice="test_female", name="long")
    assert result.segments[0].chunks > 1          # 内部分块
    assert len(result.segments) == 1              # 对外仍是一段
    assert Path(result.path).is_file()


def test_prepare_can_be_disabled_for_ab(module) -> None:
    with_prepare = module.synthesize(
        SynthRequest(text="详情见 https://qwen.ai 谢谢。", voice="test_male"), save=False)
    without = module.synthesize(
        SynthRequest(text="详情见 https://qwen.ai 谢谢。", voice="test_male",
                     prepare_text=False), save=False)
    # 关掉预处理后 URL 还在，文本更长 → sine 引擎的时长更长
    assert without.seconds > with_prepare.seconds
    assert "qwen" not in with_prepare.segments[0].spoken_text


def test_runaway_is_retried_with_a_new_seed(module) -> None:
    result = module.synthesize(
        SynthRequest(text="这一句会先失控一次然后成功。", voice="test_male",
                     gen={"_simulate_runaway": 1}), save=False)
    assert result.ok
    assert result.segments[0].retries == 1


def test_unrecoverable_runaway_raises_with_the_reason(module) -> None:
    """救不回来时不能把噪声当成品交出去。"""
    with pytest.raises(SynthError, match="撞上限"):
        module.synthesize(SynthRequest(text="这一句一直失控。", voice="test_male",
                                       gen={"_simulate_runaway": 9}), save=False)


def test_failed_segment_is_excluded_from_the_merge(module) -> None:
    module.config.guard.retry = 0
    request = BatchSynthRequest(segments=[
        SynthRequest(text="这一段是好的，可以正常合成。", voice="test_female"),
        SynthRequest(text="这一段一直失控救不回来。", voice="test_male",
                     gen={"_simulate_runaway": 9}),
    ], merge=True, name="partial")
    result = module.synthesize_many(request)
    assert not result.ok                                  # 整体不算成功
    assert result.segments[1].failed
    assert any("未计入合并音频" in w for w in result.warnings)
    # 合并音频只包含好的那一段
    assert result.seconds == pytest.approx(result.segments[0].seconds, abs=0.05)


def test_batch_writes_per_segment_files_and_a_merged_one(module) -> None:
    result = module.synthesize_many(BatchSynthRequest(segments=[
        SynthRequest(text="旁白的第一句话在这里。", voice="test_female", role="旁白"),
        SynthRequest(text="记者的第二句话在这里。", voice="test_male", role="记者"),
    ], merge=True, name="batch"))
    assert all(Path(s.path).is_file() for s in result.segments)
    assert Path(result.path).name == "merged.wav"
    manifest = json.loads((Path(result.output_dir) / "manifest.json").read_text("utf-8"))
    assert manifest["ok"] and len(manifest["segments"]) == 2
    # manifest 里不该出现 numpy 波形（序列化会炸）
    assert "wav" not in manifest["segments"][0]


def test_role_change_uses_a_longer_gap(module) -> None:
    module.config.audio.pause_ms = 100
    module.config.audio.role_change_pause_ms = 900
    same = module.synthesize_many(BatchSynthRequest(segments=[
        SynthRequest(text="第一句话在这里。", voice="test_female", role="旁白"),
        SynthRequest(text="第二句话在这里。", voice="test_female", role="旁白"),
    ], merge=True, save_segments=False, name="same"))
    switched = module.synthesize_many(BatchSynthRequest(segments=[
        SynthRequest(text="第一句话在这里。", voice="test_female", role="旁白"),
        SynthRequest(text="第二句话在这里。", voice="test_male", role="记者"),
    ], merge=True, save_segments=False, name="switched"))
    assert switched.seconds == pytest.approx(same.seconds + 0.8, abs=0.05)


def test_explicit_pause_ms_wins(module) -> None:
    module.config.audio.pause_ms = 100
    result = module.synthesize_many(BatchSynthRequest(segments=[
        SynthRequest(text="第一句话在这里。", voice="test_female", pause_ms=1000),
        SynthRequest(text="第二句话在这里。", voice="test_female"),
    ], merge=True, save_segments=False, name="explicit"))
    total = sum(s.seconds for s in result.segments)
    assert result.seconds == pytest.approx(total + 1.0, abs=0.05)


def test_segment_seed_is_stable_per_index(module) -> None:
    """
    单段重生成不能影响其他段 —— GUI「只重做第 3 句」靠的就是这条。
    """
    first = module.synthesize_many([
        SynthRequest(text="第一句话在这里。", voice="test_female"),
        SynthRequest(text="第二句话在这里。", voice="test_female"),
    ], merge=False, save_segments=False)
    again = module.synthesize_many([
        SynthRequest(text="第一句话在这里。", voice="test_female"),
        SynthRequest(text="第二句话在这里。", voice="test_female"),
    ], merge=False, save_segments=False)
    assert [s.seed for s in first.segments] == [s.seed for s in again.segments]
    assert first.segments[0].seed != first.segments[1].seed


def test_execution_is_grouped_by_checkpoint(module) -> None:
    """
    8 GB 卡上换 checkpoint = 卸载 + 重载几十秒。混模式的多段必须按模式分组，
    否则轮动次数等于段数而不是模式数。**结果顺序仍是原始顺序。**
    """
    modes = [Mode.CUSTOM_VOICE, Mode.VOICE_DESIGN, Mode.CUSTOM_VOICE, Mode.VOICE_DESIGN]
    order = module._execution_order(modes)
    assert order == [0, 2, 1, 3]

    result = module.synthesize_many([
        SynthRequest(text="内置音色的第一段。", voice="test_female"),
        SynthRequest(text="音色设计的第二段。", mode=Mode.VOICE_DESIGN, instruct="沉稳女声"),
        SynthRequest(text="内置音色的第三段。", voice="test_female"),
    ], merge=False, save_segments=False)
    assert [s.index for s in result.segments] == [0, 1, 2]      # 输出顺序不变


def test_no_grouping_when_all_modes_fit(module) -> None:
    module.config.engine.max_resident = 3
    modes = [Mode.CUSTOM_VOICE, Mode.VOICE_DESIGN, Mode.CUSTOM_VOICE]
    assert module._execution_order(modes) == [0, 1, 2]          # 保持原顺序，日志好读


def test_unsupported_mode_fails_loudly(module) -> None:
    """能力不足要报错，不能静默降级成别的模式。"""
    module.engine.capabilities = lambda: {Capability.CUSTOM_VOICE}
    with pytest.raises(CapabilityError, match="voice_design"):
        module.synthesize(SynthRequest(text="换个音色试试。", mode=Mode.VOICE_DESIGN,
                                       instruct="沉稳女声"), save=False)


def test_split_then_batch(module) -> None:
    """GUI 的主流程：一整段文本 → 拆条 → 全部合成。"""
    pieces = module.split("旁白：今天的资讯。\n记者：第一条消息。", rule="role")
    result = module.synthesize_many([
        SynthRequest(text=p.text, role=p.role, voice=p.voice or "test_female")
        for p in pieces], merge=True, name="pipeline")
    assert result.ok and len(result.segments) == 2
    assert [s.role for s in result.segments] == ["旁白", "记者"]


def test_merge_audio_reuses_files_without_calling_the_engine(module) -> None:
    """
    合并已生成的段落**不能重跑模型**。

    重跑既慢（一段几十秒），又因为解码是采样的而会给出和刚试听过不一样的音频 ——
    「明明听着好，一合并就变了」是最难解释的一种问题。
    这里用引擎调用次数把它钉死。
    """
    first = module.synthesize("第一段的内容在这里。", voice="test_female", name="reuse")
    second = module.synthesize("第二段的内容在这里。", voice="test_male", name="reuse")

    calls = {"n": 0}
    original = module.engine.synthesize

    def counting(chunk):
        calls["n"] += 1
        return original(chunk)

    module.engine.synthesize = counting
    merged = module.merge_audio([first.path, second.path], gaps_ms=[500, 0], name="reuse")

    assert calls["n"] == 0                      # ← 一次都没调引擎
    assert Path(merged["path"]).is_file()
    assert merged["count"] == 2
    assert merged["seconds"] == pytest.approx(first.seconds + second.seconds + 0.5, abs=0.05)


def test_merge_audio_rejects_missing_file(module) -> None:
    """少一段音频要明确报错（并提示"这一段可能还没生成"），不能悄悄少拼一段。"""
    ok = module.synthesize("有音频的这一段。", voice="test_female", name="miss")
    with pytest.raises(SynthError, match="不存在"):
        module.merge_audio([ok.path, str(Path(ok.path).parent / "nope.wav")], name="miss")


def test_progress_hook_reports_every_segment(module) -> None:
    """
    界面靠这个回调知道"跑到第几段了" —— 一条几十秒，没有它只能盯着不动的界面猜。
    事件顺序按**执行顺序**（可能被 checkpoint 分组重排），不是原始顺序。
    """
    events: list[tuple] = []
    module.synthesize_many(
        [SynthRequest(text="第一段内容。", voice="test_female"),
         SynthRequest(text="第二段内容。", voice="test_male")],
        merge=False, save_segments=False,
        progress=lambda event, index, total, result: events.append((event, index, total)))

    assert [e[0] for e in events] == ["start", "done", "start", "done"]
    assert {e[1] for e in events} == {0, 1}
    assert all(e[2] == 2 for e in events)


def test_new_run_name_is_unique(module) -> None:
    """会话目录名不能撞：同一秒内取两次要给出不同的名字。"""
    first = module.new_run_name("gui")
    (module.config.output_path / first).mkdir(parents=True)
    assert module.new_run_name("gui") != first


def test_info_reports_engine_and_pipeline(module) -> None:
    info = module.info()
    assert info["backend"] == "sine"
    assert info["engine"]["capabilities"]
    assert "pause_ms" in info["audio"]
