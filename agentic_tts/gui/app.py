"""
图形界面 / GUI（NiceGUI，深色控制台风）—— 单段试听 + 多段流水线。

────────────────────────────────────────────────────────────────────────────
界面要解决的五件事
────────────────────────────────────────────────────────────────────────────
1. **随时知道程序在不在跑**：一条合成几十秒起。顶栏一颗状态灯（就绪/合成中/完成/失败，
   带秒表），多段时**正在跑的那一张卡高亮**并逐段更新状态标记。
   进度靠门面的 `progress` 回调传出来，界面侧用一个 `ui.timer` 轮询 ——
   回调在工作线程里，不能直接碰 UI。
2. **一次会话的产物都在同一个目录**：页面打开时就定下 `gui-<时间戳>/`，
   之后逐段重做、整段合并全部写在里面。否则十几段散在十几个时间戳目录里，事后找不齐。
3. **统一声音配置**：多段面板顶部一组配置（含音色克隆/音色设计），默认全段共用；
   某一段要不一样时展开「单独配置」覆盖它。
4. **控件按模式显隐**：只有选了音色克隆才出现参考音频上传（而且是压扁的一条），
   否则那个 Quasar 上传框会常驻占掉半屏。
5. **合并优先复用已有音频**：每段都已生成就**只拼接、不重跑模型** ——
   重跑既慢又会因为解码采样而给出和刚试听过不一样的音频。

⚠️ **UI 必须建在 `@ui.page('/')` 里。** NiceGUI 对"自动首页"的实现是每来一个客户端就
重跑一遍入口脚本，于是 `ui.run()` 被重复执行、页面直接 500（RAG 模块实测踩过）。
用显式 page 装饰器后每个客户端还各自拿到自己的段落状态。

⚠️ 合成必须扔到线程里（`run.io_bound`），并用 **`module.lock`** 串行 ——
同一张 8 GB 卡上并发两条必然 OOM。锁在 `TTSModule` 上而不是这个文件里，
是因为界面通常**挂在 HTTP 服务的同一个进程里**（见 `mount()`）：
两边各拿一把锁，就会在「界面点生成 + 一条 HTTP 请求」时同时要两份权重。

────────────────────────────────────────────────────────────────────────────
两种起法 / Two ways to serve this UI
────────────────────────────────────────────────────────────────────────────
- **`mount(app, module)`（推荐，默认）**：挂到 HTTP 服务上，`/gui`。
  一个进程、**一份权重**、一把锁。`cli serve` 就是这么做的。
- `run(...)`：独立进程（`cli gui`）。它会**自己再造一个 `TTSModule`** ——
  和服务一起用就是两份权重，8 GB 卡上是实打实的内存压力，所以只在
  「只要界面、不要 HTTP」时用。
  A standalone process loads a second copy of the weights; prefer mounting.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from nicegui import app as nicegui_app
from nicegui import run as nicegui_run
from nicegui import ui

from agentic_tts.core import handoff
from agentic_tts.core.errors import TTSError
from agentic_tts.core.types import (
    BatchSynthRequest,
    Capability,
    Mode,
    SplitRule,
    SynthRequest,
)
from agentic_tts.gui import theme
from agentic_tts.text.events import known_events

_MODE_LABELS = {
    Mode.CUSTOM_VOICE: "内置音色",
    Mode.VOICE_DESIGN: "音色设计",
    Mode.VOICE_CLONE: "音色克隆",
}
_MODE_CAPABILITY = {
    Mode.CUSTOM_VOICE: Capability.CUSTOM_VOICE,
    Mode.VOICE_DESIGN: Capability.VOICE_DESIGN,
    Mode.VOICE_CLONE: Capability.VOICE_CLONE,
}
_SPLIT_LABELS = {
    SplitRule.PUNCT: "按标点（一句一段）",
    SplitRule.CHARS: "按字数",
    SplitRule.ROLE: "按角色（角色：台词）",
    SplitRule.BLANK: "按空行",
}


def _handoff_token() -> str:
    """
    取 URL 上的 `?import=<token>` / Read the hand-off token off the URL.

    别的应用（如 DailyNewsAssistant）把稿子 POST 到 `/gui/handoff` 拿到 token，
    再打开 `GUI?import=<token>`，文本就直接落进多段界面 —— 不必让人复制粘贴一遍。
    取不到就当普通打开，绝不因为这件事让页面起不来。
    """
    try:
        request = ui.context.client.request
        return str(request.query_params.get("import", "")) if request else ""
    except Exception:                                 # noqa: BLE001 - 拿不到就算了
        return ""


def run(config: Optional[str] = None, engine: Optional[str] = None,
        host: Optional[str] = None, port: Optional[int] = None) -> None:
    """
    起一个**独立**的界面进程 / Start the GUI as its own process.

    ⚠️ 它会自己造一个 `TTSModule`：和 HTTP 服务一起用就是**两份权重**。
    两个都要的话用 `cli serve`（界面挂在 `/gui`，一份权重）。
    """
    from agentic_tts.engine import TTSModule

    module = TTSModule(config)
    if engine:
        module.config.engine.backend = engine

    build(module)
    cfg = module.config.gui
    ui.run(host=host or cfg.host, port=port or cfg.port, title="Agentic TTS",
           reload=False, show=False, favicon="🎙️", dark=True)


def mount(fastapi_app: Any, module: Any, *, path: str = "/gui") -> None:
    """
    把界面挂到已有的 FastAPI 应用上 / Mount the UI onto an existing FastAPI app.

    **和 HTTP 服务共用同一个 `TTSModule`**，于是：一份权重、一把锁、
    一个端口。调用方只要知道一个地址（`http://host:port/gui`）。
    Sharing the module means one copy of the weights, one lock and one port.
    """
    build(module, mount_path=path.rstrip("/"))
    ui.run_with(fastapi_app, mount_path=path, title="Agentic TTS",
                favicon="🎙️", dark=True, show_welcome_message=False)


def build(module: Any, *, mount_path: str = "") -> None:  # noqa: C901 - 界面装配
    """
    注册界面页面与静态路由 / Register the page and its static routes.

    `mount_path`: 挂载前缀。挂到 `/gui` 之后，产物的静态路由在浏览器看到的是
    `/gui/outputs/...` —— 播放器的 URL 必须带上这个前缀，否则 404。
    Media URLs need the prefix once mounted, or the players 404.
    """
    outputs = module.config.output_path
    outputs.mkdir(parents=True, exist_ok=True)
    # 产物目录挂成静态媒体，段落播放器才有 URL 可指
    nicegui_app.add_media_files("/outputs", outputs)

    capabilities = module.capabilities()
    voice_names = [v.name for v in module.voice_list()]
    native_events = Capability.VOCAL_EVENTS in capabilities
    can_instruct = Capability.INSTRUCT in capabilities
    clone_ok = Capability.VOICE_CLONE in capabilities

    def media_url(path: Optional[str]) -> Optional[str]:
        """产物绝对路径 → 浏览器可取的 URL / Absolute path → a browser-visible URL."""
        if not path:
            return None
        try:
            relative = Path(path).resolve().relative_to(outputs.resolve()).as_posix()
        except ValueError:
            return None
        return f"{mount_path}/outputs/{relative}"

    def open_folder(path: Optional[str]) -> None:
        """
        打开产物所在目录 / Reveal the output folder in the file manager.

        ⚠️ 打开的是**跑 GUI 那台机器**上的文件管理器（浏览器没有本地文件权限，
        这件事只能由服务端做）。本机使用没问题；远程访问时点了不会有反应，
        所以这里把路径也一并显示出来，随时能复制。
        Opens on the machine running the GUI, not the browser host.
        """
        if not path:
            return
        target = Path(path)
        directory = target if target.is_dir() else target.parent
        try:
            if os.name == "nt":
                os.startfile(directory)  # noqa: S606 - 本机打开资源管理器
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(directory)])
            else:
                subprocess.Popen(["xdg-open", str(directory)])
        except OSError as exc:
            ui.notify(f"打不开目录：{exc}　路径：{directory}", type="warning",
                      multi_line=True, close_button=True)

    # ------------------------------------------------------------------ 页面

    @ui.page("/")
    def index() -> None:  # noqa: C901 - 界面装配，拆开反而更难读
        theme.apply()

        # 每个客户端一份状态 / per-client state
        session = module.new_run_name("gui")      # ② 一次会话共用一个产物目录
        # `?import=<token>` 带进来的交接单（别的应用把稿子推过来，见 core/handoff.py）
        incoming = _handoff_token()
        run_state: dict[str, Any] = {"busy": False, "started": 0.0, "label": "● 就绪"}
        # 工作线程只往这里写，UI 由 timer 读 —— 跨线程不直接碰 UI
        progress_box: dict[str, Any] = {"stamp": 0, "seen": 0, "index": None,
                                        "total": 0, "event": "", "targets": []}
        segments: list[dict[str, Any]] = []

        # ---------------------------------------------------------- 顶栏

        with ui.header().classes("tt-header items-center justify-between px-4 py-2"):
            with ui.row().classes("items-center gap-3"):
                ui.html('<span class="tt-title">AGENTIC<span class="accent">·</span>TTS</span>')
                engine_info = module.info().get("engine", {})
                ui.label(f"{module.config.engine.backend} / "
                         f"{engine_info.get('device', '-')} / "
                         f"{engine_info.get('dtype', '-')} / "
                         f"驻留 {module.config.engine.max_resident}").classes("tt-meta")
            with ui.row().classes("items-center gap-3"):
                spinner = ui.spinner("dots", size="sm")
                spinner.set_visibility(False)
                status = ui.label("● 就绪").classes("tt-status")
                ui.label(f"产物 {session}").classes("tt-meta")

        def set_status(text: str, kind: str = "idle") -> None:
            """
            刷顶栏状态灯 / Update the status pill.

            `kind`: idle / run / ok / err。文本自带符号，颜色只是加强 ——
            黑白截图与色觉障碍下也要读得出来。
            """
            run_state["label"] = text
            css = {"run": "tt-status is-run", "ok": "tt-status is-ok",
                   "err": "tt-status is-err"}.get(kind, "tt-status")
            status.set_text(text)
            status.classes(replace=css)
            spinner.set_visibility(kind == "run")

        # ------------------------------------------------- 段落状态 / chips

        def segment_state(seg: dict[str, Any]) -> str:
            if seg.get("running"):
                return "running"
            if seg.get("failed"):
                return "failed"
            if not seg.get("path"):
                return "idle"
            # 生成之后又改了文本 ⇒ 手里那条音频已经对不上稿子，必须说出来，
            # 否则「合并复用已有音频」会把旧音频拼进成品。
            if (seg["body"].value or "").strip() != (seg.get("generated_text") or ""):
                return "stale"
            return "done"

        def refresh_chip(seg: dict[str, Any]) -> None:
            state = segment_state(seg)
            label, css = theme.chip(state)
            seg["chip"].set_text(label)
            seg["chip"].classes(replace=css)
            seg["card"].classes(replace={
                "running": "tt-card tt-seg is-running w-full",
                "done": "tt-card tt-seg is-done w-full",
                "failed": "tt-card tt-seg is-failed w-full",
            }.get(state, "tt-card tt-seg w-full"))

        def refresh_all_chips() -> None:
            for seg in segments:
                refresh_chip(seg)

        # ------------------------------------------------- 进度轮询 / ticker

        def tick() -> None:
            """
            0.3 秒一次：把工作线程写下的进度画到界面上 / Paint worker-thread progress.

            **秒表也在这里走** —— 合成期间界面上必须有个动的东西，
            否则「是不是卡死了」只能靠猜。
            """
            if progress_box["stamp"] != progress_box["seen"]:
                progress_box["seen"] = progress_box["stamp"]
                index, total = progress_box["index"], progress_box["total"]
                targets = progress_box.get("targets") or []
                for seg in segments:
                    seg["running"] = False
                if progress_box["event"] == "start" and index is not None:
                    if index < len(targets):
                        targets[index]["running"] = True
                    run_state["label"] = f"▶ 合成中　第 {index + 1} 段（共 {total} 段）"
                refresh_all_chips()
            if run_state["busy"]:
                status.set_text(f"{run_state['label']}　"
                                f"{time.perf_counter() - run_state['started']:5.1f}s")

        ui.timer(0.3, tick)

        # ------------------------------------------------- 合成入口 / runners

        def on_progress(event: str, index: int, total: int, result: Any) -> None:
            """门面在**工作线程**里调这个 —— 只写字典，不碰 UI。"""
            progress_box.update(event=event, index=index, total=total,
                                stamp=progress_box["stamp"] + 1)

        async def run_synth(requests: list[SynthRequest], *, merge: bool, label: str,
                            targets: Optional[list[dict[str, Any]]] = None,
                            name: Optional[str] = None):
            """跑一批合成 / Run one batch off the event loop。出错返回 None（已通知用户）。"""
            def work():
                with module.lock:
                    return module.synthesize_many(
                        BatchSynthRequest(segments=requests, merge=merge,
                                          save_segments=True, name=name),
                        progress=on_progress)

            progress_box["targets"] = targets or []
            set_status(label, "run")
            run_state.update(busy=True, started=time.perf_counter())
            try:
                return await nicegui_run.io_bound(work)
            except (TTSError, ValueError) as exc:
                set_status("▲ 失败", "err")
                ui.notify(str(exc), type="negative", multi_line=True, close_button=True)
                return None
            finally:
                run_state["busy"] = False
                for seg in segments:
                    seg["running"] = False
                refresh_all_chips()

        def report_handoff(paths: list[str], *, done: bool = True) -> None:
            """
            把进度与产物清单写回交接单 / Write progress and files back to the record.

            交接单是**跨进程**的（调用方轮询 HTTP 服务读它），所以这里只写文件，
            不做任何回调 —— 调用方可能在另一台机器上，我们连它在不在都不知道。
            The caller polls the record over HTTP; it may be on another machine.

            顺带写 `done` / `total`：调用方的等待动效就是靠这两个数字动起来的，
            否则它只能显示一个「还在等」，和卡死看起来一样。
            The counts drive the caller's progress bar; without them it can only show
            an indeterminate "still waiting", which looks like a hang.
            """
            if not incoming:
                return
            files: list[str] = []
            for path in paths:
                if not path:
                    continue
                try:                     # 相对输出根，正好是 /outputs/{run}/{file} 的形状
                    files.append(Path(path).resolve()
                                 .relative_to(outputs.resolve()).as_posix())
                except ValueError:
                    continue
            handoff.update(outputs, incoming,
                           status="done" if done else "running",
                           files=files, run=session,
                           done=sum(1 for s in segments if s.get("path")),
                           total=len(segments) or 1)

        def offer_return() -> None:
            """
            回交接方的界面 / Go back to whoever handed the script over.

            调用方在 `return_url` 里说了它自己在哪。生成完就该回去 ——
            留在这个页面上，人还得自己找回原来那个标签页。
            **优先关掉本标签页**（它是被 `window.open` 打开的，关得掉），
            这样焦点自然回到原来那一页，也不会多出第二个调用方页面；
            关不掉（比如手动粘贴 URL 打开的）再退回跳转。
            Closing this script-opened tab returns focus to the original page without
            creating a duplicate; navigation is the fallback when close is blocked.
            """
            record = handoff.read(outputs, incoming) if incoming else None
            back = (record or {}).get("return_url") or ""
            if not back:
                return

            ui.notify("产物已回传，正在返回…", type="positive")

            def go() -> None:
                ui.run_javascript("window.close()")
                ui.navigate.to(back)

            ui.timer(1.2, go, once=True)

        async def guarded(action, *, buttons: tuple = ()) -> None:
            """
            防重复点击 / Re-entry guard.

            连点两次不只是白跑一遍：两次生成会挤在同一秒建目录（曾因此互相覆盖，
            见 issues/004），且同一张 8 GB 卡上两条请求会同时要两份权重。
            所以第二次点击**直接拒掉**，不排队。
            """
            if run_state["busy"]:
                ui.notify("正在合成，请等这一条跑完（同一张卡上不能并发）", type="warning")
                return
            for button in buttons:
                button.disable()
            try:
                result = action()
                if inspect.isawaitable(result):
                    await result
            finally:
                for button in buttons:
                    button.enable()

        # ------------------------------------------------- 顶栏 / header
        # 三档引擎开关 + 状态指示 + fallback 计数
        engine_mode_state: dict[str, Any] = {"current": module.engine_mode}
        fallback_indicator = ui.label("").classes("text-xs text-orange-400 hidden")
        fallback_count_label = ui.label(f"fallback: {module._fallback_count}").classes("text-xs text-grey-5")

        def refresh_fallback_count() -> None:
            fallback_count_label.set_text(f"fallback: {module._fallback_count}")
            if engine_mode_state["current"] == "auto":
                fallback_indicator.classes(remove="hidden")
                fallback_indicator.set_text("⚡ auto 模式：local 失败会自动 fallback 到 API")
            else:
                fallback_indicator.classes(add="hidden")

        ui.timer(2.0, refresh_fallback_count)

        def set_engine_mode(mode: str) -> None:
            if mode == engine_mode_state["current"]:
                return
            engine_mode_state["current"] = mode
            module.engine_mode = mode
            _logger.info("engine mode → %s", mode)
            ui.notify(f"引擎档位：{mode}", type="info")

        with ui.row().classes("items-center gap-2 w-full"):
            ui.label("引擎").classes("text-sm text-grey-4")
            engine_btn_group = ui.button_group().props("push glossy").classes("gap-1")
            for m, label in [("local", "Local"), ("api", "API"), ("auto", "Auto (Fallback)")]:
                btn = engine_btn_group.button(label)
                if m == engine_mode_state["current"]:
                    btn.props("color=primary")
                btn.on_click(lambda _, mode=m: set_engine_mode(mode))
            ui.space()
            fallback_indicator  # 占位：fallback 触发时显示
            fallback_count_label

        async def on_upload(event: Any, state: dict[str, Any]) -> None:
            """
            存参考音频 / Persist the uploaded reference audio.

            ⚠️ 上传事件的形状跨 NiceGUI 大版本变过：3.x 是 `event.file`
            （`FileUpload`，`read()` 是 async），2.x 是 `event.name` + `event.content`。
            两种都认，否则换个版本就是 `AttributeError`（本机 3.14 上实际踩到）。
            """
            upload = getattr(event, "file", None)
            if upload is not None:                       # NiceGUI 3.x
                filename, data = upload.name, await upload.read()
            else:                                        # NiceGUI 2.x
                filename, data = event.name, event.content.read()

            target_dir = module.config.voices_path / "refs"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / Path(filename).name
            target.write_bytes(data)
            state["ref_audio"] = str(target)
            state["ref_label"].set_text(target.name)
            ui.notify(f"参考音频已存到 {target}", type="positive")

        def voice_controls() -> dict[str, Any]:
            """
            一组「模式 + 音色 + 指令 + 参考音频 + 事件」控件 / One voice-control group.

            统一配置与每段的单独配置用的是**同一份实现** —— 两处各写一份迟早不一样。

            两条界面决定：
            · 引擎不支持的模式在下拉里标注出来（能力表静态可查，开机就知道）；
            · 控件**按模式显隐**：参考音频只在音色克隆时出现。Quasar 的上传框默认是个
              带标题栏和文件列表的大盒子，常驻会占掉半屏，这里配合 `tt-upload`
              的 CSS 压成一条虚线细带。
            """
            state: dict[str, Any] = {"ref_audio": None}
            with ui.column().classes("w-full gap-2"):
                with ui.row().classes("items-center gap-2 w-full"):
                    options = {}
                    for mode, label in _MODE_LABELS.items():
                        ok = _MODE_CAPABILITY[mode] in capabilities
                        options[mode.value] = label if ok else f"{label}（不支持）"
                    state["mode"] = ui.select(
                        options,
                        value=(Mode.CUSTOM_VOICE.value
                               if Capability.CUSTOM_VOICE in capabilities
                               else next(iter(options))),
                        label="模式").props("outlined dense").classes("w-36")
                    state["voice"] = ui.select(
                        voice_names or ["（无可用音色）"],
                        value=voice_names[0] if voice_names else None,
                        label="音色", with_input=True).props("outlined dense").classes("w-44")
                    state["instruct"] = ui.input(
                        label="语气指令 / 音色描述",
                        placeholder="语速平缓，播音腔" if can_instruct else "本引擎不支持",
                    ).props("outlined dense").classes("flex-grow")
                    state["instruct"].set_enabled(can_instruct)
                    if not can_instruct:
                        state["instruct"].tooltip("当前引擎没有 instruct 能力，填了不会生效")

                with ui.row().classes("items-center gap-2 w-full") as clone_row:
                    with ui.column().classes("gap-0 w-56 tt-upload"):
                        upload = ui.upload(on_upload=lambda e: on_upload(e, state),
                                           auto_upload=True, label="参考音频 wav/mp3") \
                            .props("flat dense accept=.wav,.mp3,.flac")
                        state["ref_label"] = ui.label("未选择").classes("tt-hint")
                    state["ref_text"] = ui.input(label="参考音频原话（ICL 必填）") \
                        .props("outlined dense").classes("flex-grow")
                    state["x_vector_only"] = ui.checkbox("只用音色向量")
                    if not clone_ok:
                        for widget in (upload, state["ref_text"], state["x_vector_only"]):
                            widget.set_enabled(False)
                        upload.tooltip("当前引擎不支持音色克隆")

                with ui.row().classes("items-center gap-1 flex-wrap"):
                    ui.label("插入事件").classes("tt-hint")
                    state["event_target"] = None

                    def insert(marker: str) -> None:
                        target = state.get("event_target")
                        if target is not None:
                            target.set_value((target.value or "") + marker)

                    ui.button("[pause]", on_click=lambda: insert("[pause:400ms]")) \
                        .props("flat dense").classes("text-xs") \
                        .tooltip("切段并插入确定长度的静音 —— 与引擎无关，一定生效")
                    for event_name in known_events():
                        if event_name == "pause":
                            continue
                        button = ui.button(f"[{event_name}]",
                                           on_click=lambda e=event_name: insert(f"[{e}]")) \
                            .props("flat dense").classes("text-xs")
                        if native_events:
                            button.tooltip("交给引擎的原生事件标记")
                        elif can_instruct:
                            button.tooltip("本引擎无原生支持，会转成 instruct 提示，效果不保证")
                        else:
                            button.set_enabled(False)
                            button.tooltip("本引擎既无原生事件也无 instruct，此标记无法实现")

            def apply_mode() -> None:
                mode = Mode(state["mode"].value)
                clone_row.set_visibility(mode is Mode.VOICE_CLONE)
                state["voice"].set_visibility(mode is Mode.CUSTOM_VOICE)

            state["mode"].on_value_change(lambda _: apply_mode())
            apply_mode()
            return state

        def request_from(state: dict[str, Any], text: str, *, role: Optional[str] = None,
                         pause_ms: Optional[int] = None,
                         seed: Optional[int] = None) -> SynthRequest:
            """控件取值 → `SynthRequest`（校验交给 pydantic，报错直接显示给用户）。"""
            mode = Mode(state["mode"].value)
            return SynthRequest(
                text=text,
                mode=mode,
                voice=state["voice"].value if mode is Mode.CUSTOM_VOICE else None,
                instruct=(state["instruct"].value or None) if can_instruct else None,
                ref_audio=state.get("ref_audio") if mode is Mode.VOICE_CLONE else None,
                ref_text=(state["ref_text"].value or None) if mode is Mode.VOICE_CLONE else None,
                x_vector_only=(bool(state["x_vector_only"].value)
                               if mode is Mode.VOICE_CLONE else False),
                role=role, pause_ms=pause_ms, seed=seed)

        # ---------------------------------------------------------- 页签

        with ui.tabs().classes("w-full") as tabs:
            tab_single = ui.tab("单段生成")
            tab_multi = ui.tab("多段生成")

        with ui.tab_panels(tabs, value=tab_single).classes("w-full"):

            # -------------------------------------------------- 单段面板
            with ui.tab_panel(tab_single):
                with ui.card().classes("tt-card w-full p-4"):
                    ui.label("文本").classes("tt-card-title")
                    single_text = ui.textarea(
                        placeholder="今天的人工智能资讯，[pause:400ms]第一条……") \
                        .props("outlined dense rows=6").classes("w-full")
                with ui.card().classes("tt-card w-full p-4"):
                    ui.label("声音配置").classes("tt-card-title")
                    single_controls = voice_controls()
                    single_controls["event_target"] = single_text

                single_player = ui.audio("").classes("w-full")
                single_player.set_visibility(False)
                with ui.row().classes("items-center gap-2"):
                    single_open = ui.button(icon="folder_open",
                                            on_click=lambda: open_folder(single_state["path"])) \
                        .props("flat dense round").tooltip("打开产物所在文件夹（服务端）")
                    single_open.set_visibility(False)
                    single_info = ui.label("").classes("tt-hint tt-mono")
                single_state: dict[str, Any] = {"path": None}

                async def generate_single() -> None:
                    if not (single_text.value or "").strip():
                        ui.notify("文本是空的", type="warning")
                        return
                    try:
                        request = request_from(single_controls, single_text.value)
                    except (TTSError, ValueError) as exc:
                        ui.notify(str(exc), type="negative", multi_line=True)
                        return
                    result = await run_synth([request], merge=False, label="▶ 合成中")
                    if result is None:
                        return
                    url = media_url(result.path)
                    if url:
                        single_player.set_source(url)
                        single_player.set_visibility(True)
                    # 耗时与音频时长是两回事，必须**分开写**：
                    # 只写一个数字时会被当成"这次跑了多久"，而音频时长通常小一个数量级
                    set_status(f"● 完成　耗时 {result.elapsed:.1f}s　"
                               f"音频 {result.seconds:.2f}s　RTF {result.rtf:.1f}", "ok")
                    single_state["path"] = result.path
                    single_open.set_visibility(bool(result.path))
                    single_info.set_text(result.path or "")
                    report_handoff([result.path])
                    for warning in dict.fromkeys(w for s in result.segments for w in s.warnings):
                        ui.notify(warning, type="warning", multi_line=True, close_button=True)

                single_buttons: list = []
                single_buttons.append(ui.button(
                    "生成",
                    on_click=lambda: guarded(generate_single, buttons=tuple(single_buttons)),
                ).classes("tt-btn-main"))

            # -------------------------------------------------- 多段面板
            with ui.tab_panel(tab_multi):
                # ① 统一声音配置（含克隆 / 设计）
                with ui.card().classes("tt-card w-full p-4"):
                    with ui.row().classes("items-center justify-between w-full"):
                        ui.label("① 统一声音配置").classes("tt-card-title")
                        use_unified = ui.checkbox("所有段共用这套配置", value=True)
                    unified = voice_controls()
                    ui.label("勾选时这套配置覆盖所有段；取消后每段卡片里的「单独配置」生效。"
                             ).classes("tt-hint")

                # ② 拆条
                with ui.card().classes("tt-card w-full p-4"):
                    ui.label("② 一整段文本 → 自动拆条").classes("tt-card-title")
                    blob = ui.textarea(
                        placeholder="旁白：今天的人工智能资讯。\n记者：第一条消息是……") \
                        .props("outlined dense rows=5").classes("w-full")
                    with ui.row().classes("items-center gap-2"):
                        rule = ui.select({r.value: label for r, label in _SPLIT_LABELS.items()},
                                         value=SplitRule.PUNCT.value, label="拆分规则") \
                            .props("outlined dense").classes("w-52")
                        max_chars = ui.number(label="字数上限",
                                              value=module.config.text.max_chars,
                                              min=10, max=400) \
                            .props("outlined dense").classes("w-28")
                        ui.button("拆条", on_click=lambda: guarded(do_split)) \
                            .classes("tt-btn-alt")
                        ui.button("加一段", on_click=lambda: add_segment()).props("flat")
                        ui.button("清空", on_click=lambda: clear()).props("flat")
                        ui.label("角色→音色映射取自 configs/config.yaml 的 voices.roles") \
                            .classes("tt-hint")

                container = ui.column().classes("w-full gap-2")
                merged_player = ui.audio("").classes("w-full")
                merged_player.set_visibility(False)
                with ui.row().classes("items-center gap-2"):
                    folder_button = ui.button(
                        icon="folder_open",
                        on_click=lambda: open_folder(str(outputs / session))) \
                        .props("flat dense round").tooltip("打开本次会话的产物文件夹（服务端）")
                    summary = ui.label("").classes("tt-hint tt-mono")

                def clear() -> None:
                    segments.clear()
                    container.clear()
                    summary.set_text("")
                    merged_player.set_visibility(False)

                def add_segment(text: str = "", role: Optional[str] = None,
                                voice: Optional[str] = None) -> None:
                    index = len(segments)
                    with container:
                        with ui.card().classes("tt-card tt-seg w-full") as card:
                            with ui.row().classes("items-center justify-between w-full"):
                                with ui.row().classes("items-center gap-2"):
                                    ui.label(f"第 {index + 1} 段").classes("tt-card-title")
                                    chip = ui.label("").classes("tt-chip tt-idle")
                                    if role:
                                        ui.label(f"角色 {role}").classes("tt-hint")
                                with ui.row().classes("gap-1"):
                                    ui.button("生成本段",
                                              on_click=lambda i=index: guarded(
                                                  lambda: generate_indices([i]))).props("dense")
                                    ui.button("换种子重掷",
                                              on_click=lambda i=index: guarded(
                                                  lambda: generate_indices([i], reroll=True))) \
                                        .props("dense flat") \
                                        .tooltip("同一段种子固定，重掷才会换一版")
                                    ui.button("删除", on_click=lambda i=index: remove(i)) \
                                        .props("dense flat color=negative")
                            body = ui.textarea(value=text).props("outlined dense rows=3") \
                                .classes("w-full")
                            override = ui.expansion("单独配置（覆盖统一配置）").classes("w-full")
                            with override:
                                controls = voice_controls()
                            controls["event_target"] = body
                            if voice and voice in voice_names:
                                controls["voice"].set_value(voice)
                            override.set_visibility(not use_unified.value)
                            with ui.row().classes("items-center gap-3"):
                                pause = ui.number(label="段后停顿(ms)", value=None,
                                                  min=0, max=5000) \
                                    .props("outlined dense").classes("w-36")
                                info = ui.label("").classes("tt-hint tt-mono")
                            player = ui.audio("").classes("w-full")
                            player.set_visibility(False)

                    seg: dict[str, Any] = {
                        "card": card, "chip": chip, "body": body, "controls": controls,
                        "override": override, "pause": pause, "info": info, "player": player,
                        "role": role, "reroll": 0, "path": None, "generated_text": None,
                        "failed": False, "running": False,
                    }
                    segments.append(seg)
                    # 改了文本要立刻反映成「文本已改」，否则合并会把旧音频拼进成品
                    body.on_value_change(lambda _, s=seg: refresh_chip(s))
                    refresh_chip(seg)

                def toggle_unified() -> None:
                    for seg in segments:
                        seg["override"].set_visibility(not use_unified.value)

                use_unified.on_value_change(lambda _: toggle_unified())

                def remove(index: int) -> None:
                    """删一段：索引会变，所以整列重建（几十段的量级，重建比维护索引安全）。"""
                    kept = [(s["body"].value, s["role"], s["controls"]["voice"].value,
                             s["path"], s["generated_text"])
                            for i, s in enumerate(segments) if i != index]
                    clear()
                    for text, role, voice, path, generated in kept:
                        add_segment(text, role, voice)
                        seg = segments[-1]
                        seg.update(path=path, generated_text=generated)
                        url = media_url(path)
                        if url:
                            seg["player"].set_source(url)
                            seg["player"].set_visibility(True)
                        refresh_chip(seg)

                def do_split() -> None:
                    if not (blob.value or "").strip():
                        ui.notify("整段文本是空的", type="warning")
                        return
                    try:
                        pieces = module.split(blob.value, rule=SplitRule(rule.value),
                                              max_chars=int(max_chars.value or 60))
                    except (TTSError, ValueError) as exc:
                        ui.notify(str(exc), type="negative")
                        return
                    clear()
                    for piece in pieces:
                        add_segment(piece.text, piece.role, piece.voice)
                    summary.set_text(f"拆出 {len(pieces)} 段")
                    missing = sorted({p.role for p in pieces if p.role and not p.voice})
                    if missing:
                        ui.notify("这些角色没配音色，会用各段自己选的音色：" + "、".join(missing),
                                  type="warning", multi_line=True)

                def controls_for(seg: dict[str, Any]) -> dict[str, Any]:
                    """统一配置 or 该段的单独配置 / The unified group, or this segment's own."""
                    return unified if use_unified.value else seg["controls"]

                def gap_for(position: int, usable: list[dict[str, Any]]) -> int:
                    """
                    段后静音 / The gap after one segment（与门面同一套规则）。

                    显式填的优先；换角色用更长的停顿，否则听众分不清是换了语气还是换了人。
                    """
                    seg = usable[position]
                    explicit = seg["pause"].value
                    if explicit:
                        return int(explicit)
                    if (position + 1 < len(usable)
                            and usable[position + 1]["role"] != seg["role"]):
                        return module.config.audio.role_change_pause_ms
                    return module.config.audio.pause_ms

                async def generate_indices(indices: list[int], *, reroll: bool = False,
                                           label: Optional[str] = None) -> bool:
                    """
                    合成指定的几段 / Synthesise the given segments.

                    产物全部落在**本次会话的同一个目录**（`name=session`），
                    逐段重做覆盖该段自己的文件 —— 这正是要的效果。
                    """
                    requests: list[SynthRequest] = []
                    targets: list[dict[str, Any]] = []
                    for index in indices:
                        seg = segments[index]
                        text = (seg["body"].value or "").strip()
                        if not text:
                            continue
                        if reroll:
                            seg["reroll"] += 1
                        seed = (module.config.engine.seed + index + 7919 * seg["reroll"]
                                if seg["reroll"] else None)
                        pause = seg["pause"].value
                        try:
                            requests.append(request_from(
                                controls_for(seg), text, role=seg["role"],
                                pause_ms=int(pause) if pause else None, seed=seed))
                        except (TTSError, ValueError) as exc:
                            ui.notify(f"第 {index + 1} 段：{exc}", type="negative",
                                      multi_line=True)
                            return False
                        targets.append(seg)

                    if not requests:
                        ui.notify("没有可合成的段落（文本都是空的）", type="warning")
                        return False

                    result = await run_synth(
                        requests, merge=False, name=session, targets=targets,
                        label=label or f"▶ 合成中（{len(requests)} 段）")
                    if result is None:
                        return False

                    for segment_result in result.segments:
                        seg = targets[segment_result.index]
                        seg["failed"] = segment_result.failed
                        seg["path"] = segment_result.path
                        seg["generated_text"] = segment_result.text
                        url = media_url(segment_result.path)
                        if url:
                            seg["player"].set_source(url)
                            seg["player"].set_visibility(True)
                        seg["info"].set_text(
                            f"耗时 {segment_result.elapsed:.1f}s　"
                            f"音频 {segment_result.seconds:.2f}s　"
                            f"RTF {segment_result.rtf:.1f}"
                            + (f"　重试 {segment_result.retries}"
                               if segment_result.retries else "")
                            + f"　种子 {segment_result.seed}")
                        refresh_chip(seg)
                        for warning in dict.fromkeys(segment_result.warnings):
                            ui.notify(f"第 {segment_result.index + 1} 段：{warning}",
                                      type="warning", multi_line=True, close_button=True)

                    done = sum(1 for s in result.segments if not s.failed)
                    set_status(f"● 完成 {done}/{len(result.segments)} 段　"
                               f"耗时 {result.elapsed:.1f}s　"
                               f"音频 {sum(s.seconds for s in result.segments):.2f}s",
                               "ok" if result.ok else "err")
                    summary.set_text(f"产物目录 {result.output_dir}")
                    # 逐段生成只**报进度**：调用方在人点「全部生成 / 合并」之前
                    # 不该把半成品收走
                    report_handoff([s["path"] for s in segments if s.get("path")],
                                   done=False)
                    return result.ok

                async def generate_all() -> None:
                    if await generate_indices(list(range(len(segments)))):
                        report_handoff([s["path"] for s in segments if s.get("path")])
                        offer_return()

                async def merge_all() -> None:
                    """
                    合并成整段 / Merge into one track.

                    ⑤ **每段都已有音频就只拼接、不重跑模型。** 重跑既慢（一段几十秒），
                    又因为解码是采样的而会给出**和刚试听过不一样**的音频 ——
                    那是最让人困惑的一种"重现不了"。
                    只有还没生成、或生成后又改了文本的段落才补跑。
                    """
                    filled = [i for i, s in enumerate(segments)
                              if (s["body"].value or "").strip()]
                    if not filled:
                        ui.notify("没有可合并的段落", type="warning")
                        return

                    pending = [i for i in filled
                               if segment_state(segments[i]) in ("idle", "stale", "failed")]
                    if pending:
                        ui.notify("第 " + "、".join(str(i + 1) for i in pending)
                                  + " 段还没有可用音频（未生成或文本已改），先补跑这几段",
                                  type="info", multi_line=True)
                        if not await generate_indices(pending,
                                                      label=f"▶ 补跑 {len(pending)} 段"):
                            return

                    usable = [segments[i] for i in filled if segments[i]["path"]]
                    if not usable:
                        return
                    if len(usable) != len(filled):
                        ui.notify("仍有段落没有音频，已按现有的合并", type="warning")

                    paths = [s["path"] for s in usable]
                    gaps = [gap_for(i, usable) for i in range(len(usable))]
                    # 字幕文字用**每段的原文**（去事件标记在门面里做），不是送进模型那份
                    texts = [(s["body"].value or "").strip() for s in usable] \
                        if want_subtitles.value else None
                    module.config.output.subtitles = bool(want_subtitles.value)
                    set_status("▶ 拼接中（复用已生成音频，不重跑模型）", "run")
                    run_state.update(busy=True, started=time.perf_counter())
                    try:
                        merged = await nicegui_run.io_bound(
                            lambda: module.merge_audio(paths, gaps_ms=gaps, texts=texts,
                                                       name=session))
                    except (TTSError, ValueError) as exc:
                        set_status("▲ 合并失败", "err")
                        ui.notify(str(exc), type="negative", multi_line=True)
                        return
                    finally:
                        run_state["busy"] = False

                    url = media_url(merged["path"])
                    if url:
                        merged_player.set_source(url)
                        merged_player.set_visibility(True)
                    set_status(f"● 合并完成　音频 {merged['seconds']:.2f}s　"
                               f"{merged['count']} 段拼接", "ok")
                    summary.set_text(merged["path"]
                                     + (f"　+ 字幕 {Path(merged['subtitle_path']).name}"
                                        if merged.get("subtitle_path") else ""))
                    report_handoff([s["path"] for s in segments if s.get("path")]
                                   + [merged["path"], merged.get("subtitle_path")])
                    offer_return()

                bottom: list = []
                with ui.row().classes("items-center gap-2 q-mt-sm"):
                    want_subtitles = ui.checkbox(
                        "导出字幕 SRT", value=module.config.output.subtitles) \
                        .tooltip("合并时顺带写一份 merged.srt：一段音频一条字幕，"
                                 "时间轴含段间静音，剪映/Premiere/DaVinci 可直接导入")
                    bottom.append(ui.button(
                        "全部生成（每段单独落盘）",
                        on_click=lambda: guarded(generate_all, buttons=tuple(bottom)),
                    ).classes("tt-btn-main"))
                    bottom.append(ui.button(
                        "合并成整段（复用已生成音频）",
                        on_click=lambda: guarded(merge_all, buttons=tuple(bottom)),
                    ).classes("tt-btn-alt"))
                    # 只在「有人把稿子交过来」时出现：逐段试听调完就想回去交差，
                    # 但没点「全部生成/合并」的话调用方还在等 —— 给一条明路。
                    # Shown only for a hand-off: the caller is still waiting when the
                    # user only generated pieces one by one.
                    if incoming:
                        def hand_back() -> None:
                            paths = [s["path"] for s in segments if s.get("path")]
                            if not paths:
                                ui.notify("还没有任何音频可回传", type="warning")
                                return
                            report_handoff(paths)
                            offer_return()

                        bottom.append(ui.button("回传并返回", icon="reply",
                                                on_click=hand_back).props("flat"))
                    ui.label(f"每段一个 wav + merged.wav，全部写在 {session}/") \
                        .classes("tt-hint")

        # ------------------------------------------ 交接单导入 / hand-off import
        #
        # 放在最后：要用到上面定义的 `add_segment` / `unified` / `tabs`。
        # 找不到交接单**只提示、不报错** —— 页面本身仍然是可用的。
        if incoming:
            record = handoff.read(outputs, incoming)
            if record is None:
                ui.notify(f"交接单 {incoming} 不存在或已过期，按普通模式打开",
                          type="warning", multi_line=True)
            else:
                clear()
                for piece in (t for t in record.get("segments") or [] if t.strip()):
                    add_segment(piece)
                voice = record.get("voice")
                if voice and voice in voice_names:
                    unified["voice"].set_value(voice)
                if record.get("instruct") and can_instruct:
                    unified["instruct"].set_value(record["instruct"])
                tabs.set_value(tab_multi)
                source = record.get("source") or "外部应用"
                title = record.get("title") or ""
                summary.set_text(f"来自 {source}　{title}".strip())
                ui.notify(f"已导入 {len(segments)} 段（来自 {source}）　"
                          f"生成完成后产物清单会自动回传",
                          type="positive", multi_line=True)


__all__ = ["build", "mount", "run"]
