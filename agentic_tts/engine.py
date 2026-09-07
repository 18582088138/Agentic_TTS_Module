"""
门面 / Facade —— `TTSModule` 是本模块唯一需要记住的类。

────────────────────────────────────────────────────────────────────────────
它负责的四件"与引擎无关"的事
────────────────────────────────────────────────────────────────────────────
1. **文本链路**：预处理 → 事件标记 → 分段（每一步都能从配置关掉）
2. **失控兜底**：撞时长上限就换种子重试，失败的段不混进成品
3. **拼接**：段间插确定长度的静音，换角色时更长；跨引擎统一采样率
4. **轮动加载排序**：多段混用不同模式时，**按 checkpoint 分组执行** ——
   8 GB 卡上换一次 checkpoint 是「卸载 + 重载几十秒」，
   不分组会来回轮动 N 次，N 是段数而不是模式数。
   Grouping by checkpoint turns N reloads into (number of distinct modes) reloads.

用法 / Usage::

    from agentic_tts import TTSModule

    tts = TTSModule()                                  # 读 configs/config.yaml
    result = tts.synthesize("今天的人工智能资讯", voice="Serena")
    print(result.path, result.seconds, result.rtf)
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from agentic_tts.audio import guard as guard_mod
from agentic_tts.audio import io as audio_io
from agentic_tts.audio import subtitle
from agentic_tts.core.config import Config
from agentic_tts.core.errors import SynthError
from agentic_tts.core.logging import get_logger
from agentic_tts.core.registry import create_engine, engine_specs
from agentic_tts.core.types import (
    BatchSynthRequest,
    Capability,
    Mode,
    SegmentResult,
    SplitRequest,
    SynthRequest,
    SynthResult,
    VoiceInfo,
)
from agentic_tts.engines.base import SynthChunk, TTSEngine
from agentic_tts.text import options_from_config, parse_events, prepare, segment as segment_text
from agentic_tts.text.events import strip_events
from agentic_tts.text.split import TextPiece, split_text, unknown_roles
from agentic_tts.voices.store import ResolvedVoice, VoiceStore

_logger = get_logger("module")

# 进度回调：(事件, 段序号, 总段数, 该段结果或 None)。事件为 "start" / "done"。
ProgressHook = Callable[[str, int, int, Optional[SegmentResult]], None]


@dataclass
class _Chunk:
    """一段请求内部被切出来的一块 / One internal chunk of a request."""

    text: str
    pause_after_ms: int
    instruct: Optional[str]
    warnings: list[str] = field(default_factory=list)


class TTSModule:
    """TTS 门面 / The TTS facade. 引擎与权重都是懒加载的 —— 造实例不花时间。"""

    def __init__(self, config: Config | str | Path | None = None) -> None:
        self.config = Config.load(config)
        self._engine: Optional[TTSEngine] = None
        self._voices: Optional[VoiceStore] = None

        # 合成串行化用的锁 / the lock that serialises synthesis
        #
        # 放在这里而不是各调用方各拿一把：HTTP 服务与图形界面**同进程共用一个
        # TTSModule**（GUI 挂在服务上，见 server/app.py）。各自一把锁的话，
        # 界面里点生成的同时来一条 HTTP 请求，就会同时要两份权重 —— 8 GB 卡必然 OOM，
        # 而 OOM 的报错完全看不出是这个原因。锁跟着模型走才不会漏。
        # The HTTP service and the GUI share one module in one process; separate locks
        # would let two synthesis runs demand two copies of the weights.
        self.lock = threading.Lock()

    # ------------------------------------------------------------ 组件 / parts

    @property
    def engine(self) -> TTSEngine:
        """当前引擎（首次访问时创建，**仍不加载权重**）/ The engine, created lazily."""
        if self._engine is None:
            self._engine = create_engine(self.config.engine.backend, self.config)
        return self._engine

    @property
    def voices(self) -> VoiceStore:
        if self._voices is None:
            self._voices = VoiceStore(self.config, builtin=self.engine.builtin_voices())
        return self._voices

    # ------------------------------------------------------- 查询 / inspection

    def voice_list(self) -> list[VoiceInfo]:
        """全部可选音色 / Every selectable voice."""
        return self.voices.all()

    def languages(self) -> list[str]:
        return self.engine.languages()

    def capabilities(self) -> set[Capability]:
        return self.engine.capabilities()

    def engines(self) -> list[dict]:
        """所有引擎的静态能力表 / Static capability table of every engine（GUI 置灰用）。"""
        return engine_specs()

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.config.engine.backend,
            "engine": self.engine.info(),
            "text": self.config.text.model_dump(),
            "audio": self.config.audio.model_dump(),
            "guard": self.config.guard.model_dump(),
            "voices_file": str(self.config.voices_file),
            "output_dir": str(self.config.output_path),
            "roles": self.config.voices.roles,
        }

    def release(self) -> None:
        """卸载权重、释放显存 / Unload weights and free memory."""
        if self._engine is not None:
            self._engine.release()

    def __enter__(self) -> "TTSModule":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    # ------------------------------------------------------------ 拆分 / split

    def split(self, request: SplitRequest | str, **kwargs: Any) -> list[TextPiece]:
        """
        把一整段文本拆成多段 / Split one blob into segments.

        规则见 `text/split.py`。角色规则会顺带把角色映射成音色（`voices.roles`
        与请求里的 `role_voices` 合并，请求优先）。
        """
        if isinstance(request, str):
            request = SplitRequest(text=request, **kwargs)
        roles = {**self.config.voices.roles, **request.role_voices}
        pieces = split_text(request.text, request.rule,
                            max_chars=request.max_chars, min_chars=request.min_chars,
                            role_voices=roles)
        missing = unknown_roles(pieces, roles)
        if missing:
            # 沉默地全用默认音色 = 听起来所有人都是同一个人，且没有任何报错
            _logger.warning("这些角色没配音色，将用默认音色：%s", "、".join(missing))
        return pieces

    # -------------------------------------------------------- 合成 / synthesis

    def synthesize(self, request: SynthRequest | str, *, save: bool = True,
                   name: Optional[str] = None, **kwargs: Any) -> SynthResult:
        """
        合成一段 / Synthesise one segment（内部可能再分段，对外仍是一条音频）。

        参数 / Args:
            request: `SynthRequest`，或直接给文本 + 关键字参数。
            save: 是否落盘 / write the audio to disk.
            name: 输出子目录名；None = 时间戳 / output sub-directory.

        返回 / Returns:
            `SynthResult`（`segments` 长度为 1）。
        """
        if isinstance(request, str):
            request = SynthRequest(text=request, **kwargs)
        return self.synthesize_many(
            BatchSynthRequest(segments=[request], merge=False,
                              save_segments=save, name=name))

    def synthesize_many(self, request: BatchSynthRequest | list[SynthRequest],
                        *, progress: Optional[ProgressHook] = None,
                        **kwargs: Any) -> SynthResult:
        """
        多段合成 / Multi-segment synthesis.

        每段可各带自己的音色/模式/指令/事件；`merge=True` 时额外产出拼接好的整段。
        **按 checkpoint 分组执行**，但结果与落盘顺序仍是原始顺序。

        参数 / Args:
            request: `BatchSynthRequest`，或一个 `SynthRequest` 列表。
            progress: 进度回调 `(事件, 段序号, 总段数, 结果)`，事件为
                `"start"` / `"done"`。**为界面而存在**：一条几十秒，
                没有它调用方只能盯着不动的界面猜程序死没死。
                回调在合成线程里被调用，界面侧不要在回调里直接改 UI。
                Called from the worker thread; the caller must not touch UI in it.

        返回 / Returns:
            `SynthResult`。整体成败看 `result.ok`；失败的段在
            `segments[i].failed` 上，其音频**不会**混进合并结果。

        抛出 / Raises:
            SynthError: 全部段都失败（一条都没成功时没有可交付的东西）。
        """
        if isinstance(request, list):
            request = BatchSynthRequest(segments=request, **kwargs)

        started = time.perf_counter()
        out_dir = self._output_dir(request.name) if (request.save_segments or request.merge) else None

        # 1) 先把每段的音色解出来 —— 分组要用到 mode，而 mode 只有解完才知道
        resolved = [self.voices.resolve(seg) for seg in request.segments]
        order = self._execution_order([r.mode for r in resolved])

        results: list[Optional[SegmentResult]] = [None] * len(request.segments)
        total = len(order)
        for index in order:
            if progress:
                progress("start", index, total, None)
            results[index] = self._synthesize_one(
                request.segments[index], resolved[index], index,
                out_dir if request.save_segments else None)
            if progress:
                progress("done", index, total, results[index])

        segments = [r for r in results if r is not None]
        if not any(not s.failed for s in segments):
            raise SynthError(
                "所有段都失败了　"
                + "；".join(w for s in segments for w in s.warnings) or "（无更多信息）")

        result = self._assemble(segments, request, out_dir, time.perf_counter() - started)
        if out_dir is not None:
            (out_dir / "manifest.json").write_text(
                json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    # ---------------------------------------------------------- 合并 / merging

    def new_run_name(self, prefix: str = "run") -> str:
        """
        取一个还没被占用的产物目录名 / Reserve an unused output directory name.

        **给界面用**：一次会话里所有段落（包括逐段重做）都该落在**同一个目录**里，
        否则一篇十几段的稿子会散在十几个时间戳目录下，事后根本找不齐。
        The GUI keeps one session directory so a dozen segments do not scatter across a
        dozen timestamped folders.
        """
        base = f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"
        for suffix in range(1, 1000):
            name = base if suffix == 1 else f"{base}-{suffix}"
            if not (self.config.output_path / name).exists():
                return name
        raise SynthError(f"同一秒内目录名冲突超过 1000 次：{base}")

    def merge_audio(self, paths: list[str | Path], *, gaps_ms: Optional[list[int]] = None,
                    texts: Optional[list[str]] = None, name: Optional[str] = None,
                    filename: str = "merged.wav") -> dict[str, Any]:
        """
        把**已有的**音频文件拼成一条 / Concatenate audio files that already exist.

        **不重新合成。** 每段都已经有音频时，合并就只是拼接 —— 再跑一遍模型既慢
        （一段几十秒）又会因为采样而得到**和你刚试听过的不一样**的音频，
        那是最让人困惑的一种"重现不了"。
        Never re-synthesises: rerunning the model would be slow and, because decoding
        samples, would yield audio different from what the user just auditioned.

        参数 / Args:
            paths: 按朗读顺序给的音频文件。
            gaps_ms: 每段**之后**的静音毫秒；缺省用配置的 `audio.pause_ms`，
                最后一项被忽略（整条结尾不该拖静音）。
            texts: 每段的字幕文字；给了且 `output.subtitles` 为真时，
                顺带写一份与拼接时间轴对齐的 SRT。
            name: 输出目录名；None = 落在自动命名的新目录。
            filename: 输出文件名。

        返回 / Returns:
            `{"path":…, "seconds":…, "sample_rate":…, "count":…}`

        抛出 / Raises:
            SynthError: 没给文件、文件不存在，或读不出来。
        """
        if not paths:
            raise SynthError("没有可合并的音频")
        clips, rates = [], []
        for item in paths:
            path = Path(item)
            if not path.is_file():
                raise SynthError(f"要合并的音频不存在：{path}　（这一段可能还没生成）")
            wav, rate = audio_io.read_wav(path)
            clips.append(wav)
            rates.append(rate)

        gaps = list(gaps_ms) if gaps_ms is not None else [self.config.audio.pause_ms] * len(clips)
        if len(gaps) != len(clips):
            raise SynthError(f"gaps_ms 长度 {len(gaps)} 与音频数 {len(clips)} 对不上")

        wav, rate = audio_io.join(clips, rates, gaps,
                                  target_rate=self.config.audio.sample_rate or rates[0])
        if self.config.audio.peak_normalize:
            wav = audio_io.peak_normalize(wav)
        out_dir = self._output_dir(name)
        path = audio_io.write_wav(out_dir / filename, wav, rate)
        subtitle_path = None
        if texts is not None:
            subtitle_path = self._write_subtitle(
                out_dir, [audio_io.duration_seconds(c, r) for c, r in zip(clips, rates)],
                texts, gaps, stem=Path(filename).stem)
        return {"path": str(path), "seconds": audio_io.duration_seconds(wav, rate),
                "sample_rate": rate, "count": len(clips), "subtitle_path": subtitle_path}

    # ------------------------------------------------------- 内部 / internals

    def _output_dir(self, name: Optional[str]) -> Path:
        """
        产物目录 / The output directory.

        ⚠️ **自动命名必须保证唯一。** 时间戳只有秒级精度，而段文件名只由「段序号 +
        音色」决定 —— 同一秒内的两次生成会落进同一个目录、写同一个文件名，
        于是后一次**静默覆盖**前一次，看起来就是「点了生成却没有新文件」。
        实测踩到（GUI 连点两次、以及 sine 引擎毫秒级返回时）。
        Auto-named runs must not collide: the timestamp has one-second resolution and
        segment filenames repeat, so a second run would silently overwrite the first.

        显式给了 `name` 则**照用不改**（`-n pytest_real` 这类是明确要求写到固定目录，
        覆盖是调用方自己的选择）。
        """
        root = self.config.output_path
        if name:
            directory = root / name
            directory.mkdir(parents=True, exist_ok=True)
            return directory

        base = time.strftime("%Y%m%d-%H%M%S")
        for suffix in range(1, 1000):
            candidate = root / (base if suffix == 1 else f"{base}-{suffix}")
            try:
                # exist_ok=False：让文件系统本身做互斥，多进程同时跑也不会撞
                candidate.mkdir(parents=True)
                return candidate
            except FileExistsError:
                continue
        raise SynthError(f"同一秒内产物目录冲突超过 1000 次：{root / base}")

    def _execution_order(self, modes: list[Mode]) -> list[int]:
        """
        执行顺序 / Execution order —— 按 checkpoint 分组，减少轮动次数。

        只在「驻留上限装不下所有模式」时才重排；否则保持原顺序（日志更好读）。
        重排是稳定的：同一模式内仍按原始顺序，于是同一段文本每次跑出来的种子一样。
        Stable within a mode, so per-segment seeds stay reproducible.
        """
        distinct = list(dict.fromkeys(modes))
        if len(distinct) <= 1 or self.config.engine.max_resident >= len(distinct):
            return list(range(len(modes)))
        _logger.info("多段混用 %d 种模式而驻留上限为 %d，按 checkpoint 分组执行以减少轮动",
                     len(distinct), self.config.engine.max_resident)
        return sorted(range(len(modes)), key=lambda i: (distinct.index(modes[i]), i))

    def _plan(self, req: SynthRequest, voice: ResolvedVoice) -> tuple[list[_Chunk], list[str]]:
        """
        文本链路 / The text pipeline —— **事件标记 → 预处理 → 分段**。

        ⚠️ **顺序不能反。** 预处理放前面会把标记本身改掉：实测
        `[pause:500ms]` 被数字规范化写成 `[pause:五百毫秒]`，于是停顿悄悄消失、
        方括号原文还留在稿子里被念出来。Markdown/URL 规则同理会啃掉标记。
        Markup must be parsed first: the number normaliser rewrites `500ms` inside it.

        返回 / Returns:
            `(chunks, warnings)`。每块自带「块后停顿」与「已折进 instruct 的事件提示」。
        """
        cfg = self.config
        warnings: list[str] = []

        use_events = req.events if req.events is not None else cfg.text.events
        if use_events:
            native = self.engine.supports(Capability.VOCAL_EVENTS)
            pieces, event_warnings = parse_events(
                req.text, default_pause_ms=cfg.audio.event_pause_ms, native=native)
            warnings.extend(event_warnings)
        else:
            from agentic_tts.text.events import EventPiece

            pieces = [EventPiece(text=req.text)]

        if req.prepare_text if req.prepare_text is not None else cfg.text.prepare:
            options = options_from_config(cfg.text)
            for piece in pieces:
                piece.text = prepare(piece.text, options)
            if not any(p.text.strip() for p in pieces):
                warnings.append("文本预处理后为空（整段可能只有 URL 或 Markdown 标记）")
            pieces = [p for p in pieces if p.text.strip()]

        use_segment = req.segment if req.segment is not None else cfg.text.segment
        chunks: list[_Chunk] = []
        for piece in pieces:
            hint = "；".join(piece.instruct_hints)
            instruct = "；".join(x for x in (voice.instruct, hint) if x) or None
            if instruct and not self.engine.supports(Capability.INSTRUCT):
                if hint:
                    warnings.append("本引擎不支持 instruct，事件提示无处可去，已忽略")
                instruct = None

            bodies = (segment_text(piece.text, max_chars=cfg.text.max_chars,
                                   min_chars=cfg.text.min_chars)
                      if use_segment else [piece.text])
            for offset, body in enumerate(bodies):
                last = offset == len(bodies) - 1
                chunks.append(_Chunk(
                    text=body,
                    # 块间用配置的默认停顿；事件标记里那个停顿挂在**该 piece 的最后一块**上
                    pause_after_ms=(piece.pause_after_ms if last else cfg.audio.pause_ms),
                    instruct=instruct,
                ))
        if not chunks:
            raise SynthError("这一段没有可合成的内容（预处理后为空？）")
        return chunks, warnings

    def _synthesize_one(self, req: SynthRequest, voice: ResolvedVoice, index: int,
                        out_dir: Optional[Path]) -> SegmentResult:
        """一段（可能内部多块）的完整流程 / One segment end to end."""
        self.engine.require_mode(voice.mode)
        chunks, warnings = self._plan(req, voice)

        # 段种子固定为 base + 段序号：**单段重生成不影响其他段**，
        # 这是 GUI「只重做第 3 句」能用的前提。
        base_seed = req.seed if req.seed is not None else self.config.engine.seed + index

        clips: list[np.ndarray] = []
        rates: list[int] = []
        gaps: list[int] = []
        retries = 0
        failed = False
        effective: dict[str, Any] = {}
        instruct_applied = False
        spoken: list[str] = []
        synth_seconds = 0.0

        for offset, chunk in enumerate(chunks):
            started = time.perf_counter()
            raw, used_retries, verdict = self._synthesize_chunk(
                chunk, voice, base_seed + offset * 1_000, req.gen)
            synth_seconds += time.perf_counter() - started
            retries += used_retries
            spoken.append(chunk.text)

            if raw is None:
                failed = True
                warnings.append(f"第 {offset + 1} 块失败并跳过：{verdict}")
                continue
            warnings.extend(raw.warnings)
            effective = raw.effective_gen
            instruct_applied = instruct_applied or raw.instruct_applied
            clips.append(raw.wav)
            rates.append(raw.sample_rate)
            gaps.append(chunk.pause_after_ms)

        if not clips:
            return SegmentResult(
                index=index, text=req.text, spoken_text="".join(spoken), mode=voice.mode,
                voice=voice.voice, role=req.role, seconds=0.0, sample_rate=0, rtf=float("inf"),
                seed=base_seed, elapsed=synth_seconds, chunks=len(chunks),
                retries=retries, failed=True, instruct=voice.instruct, warnings=warnings)

        target = self.config.audio.sample_rate or rates[0]
        wav, rate = audio_io.join(clips, rates, gaps, target_rate=target)
        seconds = audio_io.duration_seconds(wav, rate)

        path = None
        if out_dir is not None:
            stem = f"seg_{index + 1:03d}" + (f"_{voice.voice}" if voice.voice else "")
            path = str(audio_io.write_wav(out_dir / f"{stem}.{self.config.audio.format}",
                                          wav, rate))

        return SegmentResult(
            index=index, text=req.text, spoken_text="".join(spoken), mode=voice.mode,
            voice=voice.voice, role=req.role, seconds=seconds, sample_rate=rate,
            rtf=(synth_seconds / seconds if seconds else float("inf")),
            seed=base_seed, elapsed=synth_seconds, chunks=len(chunks),
            retries=retries, failed=failed,
            effective_gen=effective, instruct=voice.instruct,
            instruct_applied=instruct_applied, warnings=warnings, path=path, wav=wav)

    def _synthesize_chunk(self, chunk: _Chunk, voice: ResolvedVoice, seed: int,
                          gen: dict[str, Any]) -> tuple[Any, int, str]:
        """
        合成一块，带失控重试 / Synthesise one chunk with runaway retries.

        返回 / Returns:
            `(RawAudio | None, 重试次数, 失败原因)`。

        失控的处置是**换种子重试**，不是关采样 —— 实测贪心解码更容易失控。
        """
        cfg = self.config.guard
        attempts = cfg.retry + 1 if cfg.enabled else 1
        reason = ""

        for attempt in range(attempts):
            used_seed = seed if attempt == 0 else guard_mod.next_seed(seed, attempt)
            raw = self.engine.synthesize(SynthChunk(
                text=chunk.text, mode=voice.mode, language=voice.language,
                speaker=voice.speaker, instruct=chunk.instruct,
                ref_audio=voice.ref_audio, ref_text=voice.ref_text,
                x_vector_only=voice.x_vector_only, gen=dict(gen), seed=used_seed))
            if not cfg.enabled:
                return raw, attempt, ""

            verdict = guard_mod.check_runaway(
                raw.seconds, chunk.text,
                cap_seconds=self.engine.duration_cap_seconds(gen),
                cap_ratio=cfg.cap_ratio, chars_per_sec=cfg.chars_per_sec,
                duration_ratio=cfg.duration_ratio)
            if not verdict:
                return raw, attempt, ""
            reason = verdict.reason
            _logger.warning("疑似失控（%s），换种子重试 %d/%d", reason, attempt + 1, cfg.retry)

        return None, attempts - 1, reason

    def _assemble(self, segments: list[SegmentResult], request: BatchSynthRequest,
                  out_dir: Optional[Path], elapsed: float) -> SynthResult:
        """拼接与落盘 / Join and persist."""
        segments = sorted(segments, key=lambda s: s.index)
        usable = [s for s in segments if not s.failed and s.wav is not None]
        warnings = [f"第 {s.index + 1} 段失败，未计入合并音频" for s in segments if s.failed]

        merged_wav: Optional[np.ndarray] = None
        merged_path: Optional[str] = None
        subtitle_path: Optional[str] = None
        rate = usable[0].sample_rate if usable else 0
        seconds = sum(s.seconds for s in usable)

        if request.merge and usable:
            gaps = [self._gap_ms(usable, i, request) for i in range(len(usable))]
            merged_wav, rate = audio_io.join(
                [s.wav for s in usable], [s.sample_rate for s in usable], gaps,
                target_rate=self.config.audio.sample_rate or usable[0].sample_rate)
            if self.config.audio.peak_normalize:
                merged_wav = audio_io.peak_normalize(merged_wav)
            seconds = audio_io.duration_seconds(merged_wav, rate)
            if out_dir is not None:
                merged_path = str(audio_io.write_wav(
                    out_dir / f"merged.{self.config.audio.format}", merged_wav, rate))
                subtitle_path = self._write_subtitle(
                    out_dir, [s.seconds for s in usable],
                    [s.text for s in usable], gaps)
        elif len(usable) == 1:
            # 单段：合并结果就是它自己，`path` 直接指过去，调用方不必分两种情况处理
            merged_wav = usable[0].wav
            merged_path = usable[0].path

        return SynthResult(
            segments=segments, sample_rate=rate, seconds=seconds,
            rtf=(elapsed / seconds if seconds else float("inf")),
            elapsed=elapsed,
            engine=self.engine.info(), warnings=warnings,
            path=merged_path, subtitle_path=subtitle_path,
            output_dir=str(out_dir) if out_dir else None, wav=merged_wav)

    def _write_subtitle(self, out_dir: Path, durations: list[float], texts: list[str],
                        gaps_ms: list[int], stem: str = "merged") -> Optional[str]:
        """
        写字幕 / Write the subtitle file（一段音频一条字幕）。

        文字用**原文去掉事件标记**，不是送进模型的那份 —— 预处理会删 URL、
        把数字展开成读法，那是给模型看的，显示给人看会很怪。
        时间轴含段间静音，否则越往后字幕越提前。
        """
        if not self.config.output.subtitles or not durations:
            return None
        if self.config.output.subtitle_format != "srt":
            _logger.warning("只支持 srt 字幕，配置里写的是 %s，已跳过",
                            self.config.output.subtitle_format)
            return None
        cues = subtitle.build_cues(durations, [strip_events(t) for t in texts], gaps_ms)
        if not cues:
            return None
        return str(subtitle.write_srt(out_dir / f"{stem}.srt", cues))

    def _gap_ms(self, usable: list[SegmentResult], position: int,
                request: BatchSynthRequest) -> int:
        """
        段间静音 / The gap after one segment.

        换角色要用更长的停顿 —— 否则听众分不清是同一个人换了语气还是换了人。
        请求里显式给的 `pause_ms` 优先。
        """
        segment = usable[position]
        if segment.index < len(request.segments):
            explicit = request.segments[segment.index].pause_ms
            if explicit is not None:
                return explicit
        if position + 1 < len(usable) and usable[position + 1].role != segment.role:
            return self.config.audio.role_change_pause_ms
        return self.config.audio.pause_ms


__all__ = ["TTSModule"]
