"""
命令行 / Command line —— `tts <命令>`（或 `python -m agentic_tts.cli`）。

    tts doctor                     # 体检：配置、权重、设备、导入来源
    tts engines                    # 各引擎能力表（不加载任何权重）
    tts voices                     # 可选音色
    tts speak "今天的人工智能资讯" --voice Serena
    tts design "沉稳的女声" --text "开始播报"
    tts clone --text "开始播报" --ref-audio voices/refs/a.wav --ref-text "参考原话"
    tts split article.txt --rule role
    tts batch segments.yaml
    tts serve / tts gui

命令与 `TTSModule` 的方法一一对应 —— CLI 不做业务，只做参数解析与展示。
The CLI holds no logic; every command maps onto one facade method.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from agentic_tts.core.config import Config
from agentic_tts.core.errors import TTSError
from agentic_tts.core.types import BatchSynthRequest, Mode, SplitRule, SynthRequest

app = typer.Typer(add_completion=False, help="Agentic TTS —— 本地 TTS 服务模块")
console = Console()

_CONFIG = typer.Option(None, "--config", "-c", help="配置文件路径（默认 configs/config.yaml）")
_ENGINE = typer.Option(None, "--engine", "-e", help="临时覆盖引擎：qwen3 | sine")


def _module(config: Optional[str], engine: Optional[str] = None):
    from agentic_tts.engine import TTSModule

    module = TTSModule(config)
    if engine:
        module.config.engine.backend = engine
    return module


def _fail(exc: Exception) -> None:
    """把异常变成一行人话 + 非零退出码 / One readable line and a non-zero exit."""
    console.print(f"[red]✗[/red] {exc}")
    raise typer.Exit(code=1)


def _show(result) -> None:  # noqa: ANN001
    table = Table(show_header=True, header_style="bold")
    # 「音频」与「耗时」分两列：只给一个数字会被读成"这次跑了多久"，
    # 而 CPU 上 RTF ≈ 13，两者差一个数量级。
    for column in ("段", "音色", "模式", "音频", "耗时", "RTF", "块", "重试", "文件"):
        table.add_column(column)
    for seg in result.segments:
        table.add_row(
            str(seg.index + 1), str(seg.voice or "-"), seg.mode.value,
            f"{seg.seconds:.2f}s", f"{seg.elapsed:.1f}s", f"{seg.rtf:.1f}",
            str(seg.chunks), str(seg.retries),
            ("[red]失败[/red]" if seg.failed else (Path(seg.path).name if seg.path else "-")))
    console.print(table)
    for warning in dict.fromkeys([w for s in result.segments for w in s.warnings] + result.warnings):
        console.print(f"[yellow]![/yellow] {warning}")
    console.print(f"音频 {result.seconds:.2f}s，耗时 {result.elapsed:.1f}s，RTF {result.rtf:.1f}"
                  + (f"　→ {result.path}" if result.path else "")
                  + (f"　+ 字幕 {Path(result.subtitle_path).name}"
                     if result.subtitle_path else ""))


# --------------------------------------------------------------------- 体检


@app.command()
def doctor(config: Optional[str] = _CONFIG, engine: Optional[str] = _ENGINE) -> None:
    """
    体检 / Health check —— 配置、权重、设备、**实际导入的 qwen_tts 是哪一份**。

    最后一项是刻意查的：环境里那份 editable 安装指向已删除的目录，且磁盘上有多份
    commit 不同的源码，靠 cwd 碰运气会静默导入错的那一份而没有任何提示。
    """
    ok = True
    try:
        cfg = Config.load(config)
    except TTSError as exc:
        _fail(exc)
        return
    if engine:
        cfg.engine.backend = engine

    console.print(f"[bold]配置[/bold]　{config or '(默认) configs/config.yaml'}")
    console.print(f"引擎 {cfg.engine.backend}　设备 {cfg.engine.device}　精度 {cfg.engine.dtype}"
                  f"　驻留上限 {cfg.engine.max_resident}")

    # -- 权重 / checkpoints
    if cfg.engine.backend.startswith("qwen3"):
        for kind in cfg.engine.checkpoints:
            path = cfg.checkpoint_path(kind)
            exists = path.is_dir()
            ok &= exists
            mark = "[green]✓[/green]" if exists else "[red]✗[/red]"
            console.print(f"{mark} {kind:<13}{path}")

        # -- 源码来源 / where qwen_tts comes from
        try:
            from agentic_tts.core.config import ensure_qwen_tts_importable

            actual = ensure_qwen_tts_importable(cfg.repo_path)
            console.print(f"[green]✓[/green] qwen_tts   {actual}")
        except TTSError as exc:
            ok = False
            console.print(f"[red]✗[/red] qwen_tts   {exc}")

        # -- 设备与显存 / device and VRAM
        try:
            from agentic_tts.engines.qwen3 import (
                environment_snapshot, estimate_vram_gb, resolve_device,
                resolve_dtype_name, vram_snapshot,
            )

            device = resolve_device(cfg.engine.device)
            dtype = resolve_dtype_name(cfg.engine.dtype, device)
            vram = vram_snapshot()
            # 显存预估只在 CUDA 上有意义；CPU 上印出来会让人以为要 9.5 GB 显存
            need = (f"　预计占用显存 {estimate_vram_gb(dtype):.1f} GB"
                    if device.startswith("cuda") else "")
            console.print(f"[green]✓[/green] 实际运行  {device} / {dtype}{need}")
            if vram:
                free = vram["total_gb"] - vram["allocated_gb"]
                enough = free >= estimate_vram_gb(dtype)
                ok &= enough
                console.print(f"{'[green]✓[/green]' if enough else '[red]✗[/red]'} 显存　　　"
                              f"可用 {free:.1f} / {vram['total_gb']:.1f} GB")
            else:
                console.print("[yellow]![/yellow] 显存　　　CPU 模式（没有 CUDA 设备）")
            env = environment_snapshot()
            console.print(f"　torch {env['torch']}　transformers {env['transformers']}"
                          f"　numpy {env['numpy']}")
        except TTSError as exc:
            ok = False
            console.print(f"[red]✗[/red] 设备　　　{exc}")

    # -- 音色档案 / voice profiles
    try:
        module = _module(config, engine)
        voices = module.voice_list()
        console.print(f"[green]✓[/green] 音色　　　{len(voices)} 个"
                      f"（档案文件 {cfg.voices_file}"
                      f"{'' if cfg.voices_file.is_file() else ' —— 不存在，只用内置音色'}）")
    except TTSError as exc:
        ok = False
        console.print(f"[red]✗[/red] 音色　　　{exc}")

    console.print("[green]体检通过[/green]" if ok else "[red]体检有问题，见上面的 ✗[/red]")
    if not ok:
        raise typer.Exit(code=1)


@app.command()
def engines(config: Optional[str] = _CONFIG) -> None:
    """各引擎的静态能力表（不加载权重）/ Static capability table, no weights loaded."""
    from agentic_tts.core.registry import engine_specs

    table = Table(show_header=True, header_style="bold")
    table.add_column("引擎")
    table.add_column("状态")
    table.add_column("能力")
    table.add_column("说明")
    for spec in engine_specs():
        table.add_row(spec["name"],
                      "[green]已实现[/green]" if spec["implemented"] else "[yellow]未接入[/yellow]",
                      "、".join(spec["capabilities"]) or "-",
                      spec["description"])
    console.print(table)
    console.print("[dim]vocal_events 指**原生**事件标记；Qwen3 开源权重没有，"
                  "停顿仍确定可用（拼接层实现），其余事件转 instruct。[/dim]")


@app.command()
def voices(config: Optional[str] = _CONFIG, engine: Optional[str] = _ENGINE) -> None:
    """列出可选音色 / List selectable voices."""
    try:
        module = _module(config, engine)
        table = Table(show_header=True, header_style="bold")
        for column in ("名字", "来源", "模式", "语言", "说明"):
            table.add_column(column)
        for voice in module.voice_list():
            table.add_row(voice.name, "内置" if voice.builtin else "档案", voice.mode.value,
                          voice.language or "-", voice.description or "-")
        console.print(table)
        if module.config.voices.roles:
            console.print("角色映射：" + "、".join(f"{k}→{v}"
                                                for k, v in module.config.voices.roles.items()))
    except TTSError as exc:
        _fail(exc)


# --------------------------------------------------------------------- 合成


@app.command()
def speak(
    text: str = typer.Argument(..., help="要合成的文本（可带 [pause:400ms] / [laugh] 标记）"),
    voice: Optional[str] = typer.Option(None, "--voice", "-v", help="音色名（内置或档案）"),
    mode: Optional[str] = typer.Option(None, "--mode", "-m", help="custom_voice|voice_design|voice_clone"),
    instruct: Optional[str] = typer.Option(None, "--instruct", "-i", help="语气/语速指令"),
    language: str = typer.Option("Chinese", "--language", "-l"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="输出子目录名"),
    no_prepare: bool = typer.Option(False, "--no-prepare", help="跳过文本预处理（A/B 用）"),
    no_segment: bool = typer.Option(False, "--no-segment", help="不分段，整段直接合成"),
    config: Optional[str] = _CONFIG,
    engine: Optional[str] = _ENGINE,
) -> None:
    """合成一段文本 / Synthesise one piece of text."""
    try:
        module = _module(config, engine)
        request = SynthRequest(
            text=text, voice=voice, mode=Mode(mode) if mode else None, instruct=instruct,
            language=language, seed=seed,
            prepare_text=False if no_prepare else None,
            segment=False if no_segment else None)
        _show(module.synthesize(request, name=name))
    except (TTSError, ValueError) as exc:
        _fail(exc)


@app.command()
def design(
    instruct: str = typer.Argument(..., help="音色的自然语言描述"),
    text: str = typer.Option(..., "--text", "-t", help="要念的内容"),
    language: str = typer.Option("Chinese", "--language", "-l"),
    name: Optional[str] = typer.Option(None, "--name", "-n"),
    config: Optional[str] = _CONFIG,
    engine: Optional[str] = _ENGINE,
) -> None:
    """音色设计 / Voice design —— 用一段描述造音色（走 *-VoiceDesign 权重）。"""
    try:
        module = _module(config, engine)
        _show(module.synthesize(SynthRequest(text=text, mode=Mode.VOICE_DESIGN,
                                             instruct=instruct, language=language), name=name))
    except (TTSError, ValueError) as exc:
        _fail(exc)


@app.command()
def clone(
    text: str = typer.Option(..., "--text", "-t", help="要念的内容"),
    ref_audio: str = typer.Option(..., "--ref-audio", "-a", help="参考音频（本地文件）"),
    ref_text: Optional[str] = typer.Option(None, "--ref-text", "-r",
                                           help="参考音频里的原话（ICL 模式必须）"),
    x_vector_only: bool = typer.Option(False, "--x-vector-only",
                                       help="只用 speaker embedding，不需要 ref_text"),
    language: str = typer.Option("Chinese", "--language", "-l"),
    name: Optional[str] = typer.Option(None, "--name", "-n"),
    config: Optional[str] = _CONFIG,
    engine: Optional[str] = _ENGINE,
) -> None:
    """音色克隆 / Voice clone —— 从参考音频克隆（走 *-Base 权重）。"""
    try:
        module = _module(config, engine)
        _show(module.synthesize(SynthRequest(
            text=text, mode=Mode.VOICE_CLONE, ref_audio=ref_audio, ref_text=ref_text,
            x_vector_only=x_vector_only, language=language), name=name))
    except (TTSError, ValueError) as exc:
        _fail(exc)


@app.command()
def split(
    source: str = typer.Argument(..., help="文本文件路径，或直接给文本"),
    rule: str = typer.Option("punct", "--rule", "-r", help="chars|punct|role|blank"),
    max_chars: int = typer.Option(60, "--max-chars"),
    json_out: bool = typer.Option(False, "--json", help="输出 JSON（喂给 tts batch）"),
    config: Optional[str] = _CONFIG,
) -> None:
    """
    把一整段文本拆成多段 / Split one blob into segments.

    `--json` 的输出可以直接存成文件给 `tts batch` 用，也是 GUI 一键拆条的同一套规则。
    """
    try:
        path = Path(source)
        text = path.read_text(encoding="utf-8") if path.is_file() else source
        module = _module(config)
        pieces = module.split(text, rule=SplitRule(rule), max_chars=max_chars)
        if json_out:
            console.print_json(json.dumps(
                [{"text": p.text, "role": p.role, "voice": p.voice} for p in pieces],
                ensure_ascii=False))
            return
        table = Table(show_header=True, header_style="bold")
        for column in ("#", "角色", "音色", "字数", "文本"):
            table.add_column(column)
        for index, piece in enumerate(pieces, 1):
            table.add_row(str(index), piece.role or "-", piece.voice or "-",
                          str(len(piece.text)), piece.text[:40])
        console.print(table)
    except (TTSError, ValueError) as exc:
        _fail(exc)


@app.command()
def batch(
    source: str = typer.Argument(..., help="段落文件：.yaml/.yml/.json（段落列表）或 .txt（自动拆分）"),
    rule: str = typer.Option("punct", "--rule", "-r", help=".txt 输入时的拆分规则"),
    voice: Optional[str] = typer.Option(None, "--voice", "-v", help=".txt 输入时的默认音色"),
    merge: bool = typer.Option(True, "--merge/--no-merge", help="是否额外产出拼接好的整段"),
    name: Optional[str] = typer.Option(None, "--name", "-n"),
    config: Optional[str] = _CONFIG,
    engine: Optional[str] = _ENGINE,
) -> None:
    """
    多段合成 / Multi-segment synthesis —— 每段单独落盘，并可拼成一整条。

    段落文件格式（YAML/JSON，列表）::

        - text: "旁白第一句"
          voice: Serena
          role: 旁白
        - text: "记者追问"
          voice: Ryan
          role: 记者
          pause_ms: 800
    """
    try:
        module = _module(config, engine)
        path = Path(source)
        if path.suffix.lower() in (".yaml", ".yml", ".json"):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                _fail(ValueError(f"{path} 的顶层应是段落列表"))
                return
            segments = [SynthRequest(**item) for item in raw]
        else:
            text = path.read_text(encoding="utf-8") if path.is_file() else source
            pieces = module.split(text, rule=SplitRule(rule))
            segments = [SynthRequest(text=p.text, role=p.role, voice=p.voice or voice)
                        for p in pieces]
        console.print(f"共 {len(segments)} 段")
        _show(module.synthesize_many(BatchSynthRequest(
            segments=segments, merge=merge, save_segments=True, name=name)))
    except (TTSError, ValueError) as exc:
        _fail(exc)


# --------------------------------------------------------------- 服务与界面


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, "--host"),
    port: Optional[int] = typer.Option(None, "--port"),
    config: Optional[str] = _CONFIG,
    engine: Optional[str] = _ENGINE,
    gui: bool = typer.Option(True, "--gui/--no-gui",
                             help="把图形界面挂在 /gui（默认挂，与服务共用一份权重）"),
) -> None:
    """
    起 HTTP 服务（含图形界面）/ Start the HTTP service, GUI included.

    图形界面挂在 `/gui`，**和服务共用同一个模型**：一个进程、一份权重、
    一把锁、一个端口。另起一个 `cli gui` 进程会再加载一份权重。
    """
    import uvicorn

    from agentic_tts.server.app import create_app

    cfg = Config.load(config)
    if engine:
        cfg.engine.backend = engine
    bind_host = host or cfg.server.host
    bind_port = port or cfg.server.port
    if gui:
        console.print(f"[dim]图形界面：http://{bind_host}:{bind_port}/gui[/dim]")
    uvicorn.run(create_app(cfg, gui=gui), host=bind_host, port=bind_port)


@app.command()
def gui(
    host: Optional[str] = typer.Option(None, "--host"),
    port: Optional[int] = typer.Option(None, "--port"),
    config: Optional[str] = _CONFIG,
    engine: Optional[str] = _ENGINE,
) -> None:
    """
    只起图形界面 / Start the GUI alone（独立进程，直调门面）。

    ⚠️ **它会自己加载一份权重。** 如果 HTTP 服务也在跑，那就是两份 ——
    8 GB 卡上是实打实的内存压力。两个都要就只跑 `serve`（界面在 `/gui`）。
    """
    from agentic_tts.gui.app import run

    console.print("[yellow]提示：本命令会单独加载一份权重。"
                  "若已在跑 serve，用它的 /gui 即可（共用一份）。[/yellow]")
    run(config=config, engine=engine, host=host, port=port)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
