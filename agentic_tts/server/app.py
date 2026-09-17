"""
HTTP 服务 / HTTP service —— 端点与 `TTSModule` 的方法一一对应。

    GET  /health /info /engines /capabilities /voices
    POST /voices            新增/更新音色档案（含上传参考音频）
    POST /tts/synthesize    单段合成（可直接回 wav 二进制或 base64）
    POST /tts/design /tts/clone   语法糖，内部固定 mode
    POST /tts/batch         多段合成，每段落盘 + 可选拼接
    POST /tts/split         一整段文本 → 多段
    GET  /gui               图形界面（**挂在同一个进程里，共用一份权重**）
    POST /gui/handoff       把稿子交给 GUI 精修，回一个 token 与打开用的 URL
    GET  /gui/handoff/{token}     查这张交接单（生成完 GUI 会写回产物清单）
    GET  /outputs/{run}/{file}    取产物

⚠️ **服务内只驻留一个 `TTSModule`，且合成是串行的**（一把锁）。
不是偷懒：8 GB 卡上并发两条请求会同时要两份权重，必然 OOM，而 OOM 的报错
完全看不出是并发导致的。要并发就横向起多个进程/节点，别在单进程里放。
One module, one lock: concurrent requests would need two copies of the weights.
"""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field

from agentic_tts.audio import io as audio_io
from agentic_tts.core import handoff
from agentic_tts.core.config import Config
from agentic_tts.core.errors import CapabilityError, ConfigError, SynthError, TTSError
from agentic_tts.core.logging import get_logger
from agentic_tts.core.types import (
    BatchSynthRequest,
    Mode,
    SplitRequest,
    SynthRequest,
    SynthResult,
    VoiceInfo,
)

_logger = get_logger("server")


class SynthBody(SynthRequest):
    """单段合成的 body / Body for one-segment synthesis —— 请求字段 + 返回形态。"""

    encoding: str = Field("json", description="json（只回元信息与路径）| base64 | wav（二进制）")
    save: bool = True
    name: Optional[str] = None


class BatchBody(BatchSynthRequest):
    """多段合成的 body / Body for multi-segment synthesis."""

    encoding: str = Field("json", description="json | base64（附上合并音频）")


class VoiceBody(VoiceInfo):
    """新增/更新音色档案 / Upsert a voice profile."""


class HandoffBody(BaseModel):
    """
    交接给 GUI 的稿子 / A script handed over to the GUI.

    `segments` 直接对应多段界面的一个个文本框；只给 `text` 时当成一段。
    """

    segments: list[str] = Field(default_factory=list)
    text: str = ""
    title: str = ""
    source: str = Field("", description="谁交过来的，只用于界面提示")
    voice: Optional[str] = None
    instruct: Optional[str] = None
    return_url: str = Field("", description="生成完让界面跳回哪里（调用方自己的页面）")
    meta: dict[str, Any] = Field(default_factory=dict,
                                 description="调用方自己的上下文，原样带回")


def _http_error(exc: TTSError) -> HTTPException:
    """
    异常 → HTTP 状态码 / Map exceptions onto status codes.

    分开是为了让调用方能**自动**处置：400 改参数、409 换引擎、503 看服务端环境。
    """
    if isinstance(exc, CapabilityError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, ConfigError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, SynthError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=503, detail=str(exc))


def _payload(result: SynthResult, encoding: str) -> Any:
    """按 encoding 决定回什么 / Shape the response according to `encoding`."""
    if encoding == "wav":
        if result.wav is None:
            raise HTTPException(status_code=422, detail="没有可返回的音频（全部段失败？）")
        return Response(content=audio_io.to_wav_bytes(result.wav, result.sample_rate),
                        media_type="audio/wav")
    data = result.as_dict()
    if encoding == "base64" and result.wav is not None:
        data["audio_base64"] = base64.b64encode(
            audio_io.to_wav_bytes(result.wav, result.sample_rate)).decode("ascii")
    return data


def create_app(config: Config | str | None = None, *,
               gui: bool = True, gui_path: str = "/gui") -> FastAPI:
    """
    建 FastAPI 应用 / Build the FastAPI app（权重仍是懒加载的）。

    参数 / Args:
        gui: 是否把图形界面挂在 `gui_path` 上。**默认挂**——
            界面和服务各起一个进程就是**两份权重**，8 GB 卡上是实打实的内存压力，
            而两边要的本来就是同一个模型。挂在一起：一个进程、一份权重、一把锁、
            一个端口，调用方也只需要知道一个地址。
            Mounted by default: two processes would load two copies of the weights.
    """
    from agentic_tts.engine import TTSModule

    module = TTSModule(config)
    # 锁在 module 上：界面与 HTTP 请求共用它才能真正串行（见 engine.py）
    lock = module.lock

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        module.release()             # 退出时卸权重、清显存（仅 del 不还显存）

    app = FastAPI(title="Agentic TTS", version="0.1.0", lifespan=lifespan)
    gui_path = gui_path.rstrip("/") or "/gui"

    # ------------------------------------------------------------ 查询 / info

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "backend": module.config.engine.backend,
                "gui": gui_path if gui else ""}

    @app.get("/info")
    def info() -> dict:
        try:
            return module.info()
        except TTSError as exc:
            raise _http_error(exc) from exc

    @app.get("/engines")
    def engines() -> dict:
        return {"engines": module.engines()}

    @app.get("/api/info")
    def api_info() -> dict:
        """API 配置（api_key mask）/ API config with masked key."""
        api = module.config.api
        key = api.api_key
        masked = f"{key[:6]}***{key[-4:]}" if len(key) > 12 else ("***" if key else "")
        return {
            "enabled": api.enabled,
            "provider": api.provider,
            "endpoint": api.endpoint or "(default)",
            "model": api.model,
            "timeout_s": api.timeout_s,
            "max_retries": api.max_retries,
            "api_key_set": bool(key),
            "api_key_masked": masked,
            "voice_cache_path": api.voice_cache_path,
            "fallback_enabled": api.fallback_enabled,
            "fallback_count": module._fallback_count,
            "engine_mode": module.engine_mode,
        }

    @app.get("/api/voice-cache")
    def api_voice_cache() -> dict:
        """列出 voice_id 缓存（不含敏感信息）/ List voice_id cache."""
        from agentic_tts.engines.api.base import VoiceCache
        from pathlib import Path
        cache_path = Path(module.config.api.voice_cache_path)
        if not cache_path.is_absolute():
            from agentic_tts.core.config import PROJECT_ROOT
            cache_path = PROJECT_ROOT / cache_path
        if not cache_path.is_file():
            return {"size": 0, "entries": []}
        try:
            cache = VoiceCache(cache_path)
            entries = [
                {"key": k, "voice_id": v.get("voice_id"),
                 "provider": v.get("provider"), "ref_text": v.get("ref_text"),
                 "uploaded_at": v.get("uploaded_at")}
                for k, v in cache._data.items()  # noqa: SLF001
            ]
            return {"size": len(entries), "entries": entries}
        except Exception as exc:  # noqa: BLE001
            return {"size": 0, "entries": [], "error": str(exc)}

    @app.get("/capabilities")
    def capabilities() -> dict:
        return {"backend": module.config.engine.backend,
                "capabilities": sorted(c.value for c in module.capabilities()),
                "languages": module.languages()}

    @app.get("/voices")
    def voices() -> dict:
        try:
            return {"voices": [v.model_dump(mode="json") for v in module.voice_list()],
                    "roles": module.config.voices.roles}
        except TTSError as exc:
            raise _http_error(exc) from exc

    @app.post("/voices")
    def upsert_voice(body: VoiceBody) -> dict:
        try:
            path = module.voices.save(VoiceInfo(**body.model_dump()))
            return {"saved": body.name, "file": str(path)}
        except TTSError as exc:
            raise _http_error(exc) from exc

    @app.delete("/voices/{name}")
    def delete_voice(name: str) -> dict:
        return {"deleted": module.voices.delete(name)}

    @app.post("/voices/ref_audio")
    async def upload_ref_audio(file: UploadFile = File(...),
                               name: str = Form("")) -> dict:
        """
        上传参考音频 / Upload reference audio.

        存到 `voices/refs/` 下 —— 参考音频路径相对 `voices/` 解析，放在别处会在
        搬家时断掉（见 voices/store.py 的说明）。
        """
        target_dir = module.config.voices_path / "refs"
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = name or (file.filename or "ref.wav")
        target = target_dir / Path(filename).name
        target.write_bytes(await file.read())
        return {"path": f"refs/{target.name}", "absolute": str(target)}

    # ------------------------------------------------------- 合成 / synthesis

    @app.post("/tts/synthesize")
    def synthesize(body: SynthBody) -> Any:
        request = SynthRequest(**body.model_dump(exclude={"encoding", "save", "name"}))
        try:
            with lock:
                result = module.synthesize(request, save=body.save, name=body.name)
        except TTSError as exc:
            raise _http_error(exc) from exc
        return _payload(result, body.encoding)

    @app.post("/tts/design")
    def design(body: SynthBody) -> Any:
        body = body.model_copy(update={"mode": Mode.VOICE_DESIGN})
        return synthesize(body)

    @app.post("/tts/clone")
    def clone(body: SynthBody) -> Any:
        body = body.model_copy(update={"mode": Mode.VOICE_CLONE})
        return synthesize(body)

    @app.post("/tts/batch")
    def batch(body: BatchBody) -> Any:
        request = BatchSynthRequest(**body.model_dump(exclude={"encoding"}))
        try:
            with lock:
                result = module.synthesize_many(request)
        except TTSError as exc:
            raise _http_error(exc) from exc
        return _payload(result, body.encoding if body.encoding != "wav" else "json")

    @app.post("/tts/split")
    def split(body: SplitRequest) -> dict:
        try:
            pieces = module.split(body)
        except TTSError as exc:
            raise _http_error(exc) from exc
        return {"segments": [{"text": p.text, "role": p.role, "voice": p.voice}
                             for p in pieces]}

    # ------------------------------------------------- 交接给 GUI / GUI hand-off

    @app.post("/gui/handoff")
    def create_handoff(body: HandoffBody) -> dict:
        """
        把一份稿子交给 GUI 精修 / Hand a script over to the GUI.

        回一个 token 和现成的 URL，调用方打开它就能看到文本已经填进多段界面。
        生成完 GUI 会把 `status=done` 与产物清单写回同一条记录（见 `core/handoff.py`），
        调用方轮询 `GET /gui/handoff/{token}` 即可取回。
        """
        segments = [s for s in (body.segments or []) if s.strip()]
        if not segments and body.text.strip():
            segments = [body.text.strip()]
        if not segments:
            raise HTTPException(status_code=400, detail="交接单里没有任何文本")

        token = handoff.create(module.config.output_path, {
            "segments": segments,
            "title": body.title,
            "source": body.source,
            "voice": body.voice,
            "instruct": body.instruct,
            "return_url": body.return_url,
            "meta": body.meta,
        })
        # 界面就挂在**本服务**上（见 create_app 的 gui 参数），所以这里给的是
        # 相对路径：调用方是通过哪个地址访问到我们的，就用哪个地址打开界面。
        # 写死 host/port 会在「服务绑 0.0.0.0、别人从局域网访问」时给出一个
        # 打不开的地址。
        # A relative URL: whichever address reached us is the address that works.
        return {"token": token,
                "gui_url": f"{gui_path}/?import={token}",
                "gui_path": gui_path}

    @app.get("/gui/handoff/{token}")
    def read_handoff(token: str) -> dict:
        record = handoff.read(module.config.output_path, token)
        if record is None:
            raise HTTPException(status_code=404, detail="没有这张交接单（可能已过期）")
        return record

    # ------------------------------------------------------------ 产物 / files

    @app.get("/outputs/{run}/{filename}")
    def output_file(run: str, filename: str) -> Response:
        """
        取产物 / Fetch a produced file.

        ⚠️ 路径必须**解析后**再校验是否落在输出目录内 —— 只检查字符串会被
        `..%2f` 绕过（目录穿越）。
        The path is resolved before the containment check; a string check is bypassable.
        """
        root = module.config.output_path.resolve()
        target = (root / run / filename).resolve()
        if root not in target.parents or not target.is_file():
            raise HTTPException(status_code=404, detail="没有这个产物")
        media = "audio/wav" if target.suffix == ".wav" else "application/octet-stream"
        return Response(content=target.read_bytes(), media_type=media)

    # 界面**最后**挂：`ui.run_with` 会往 app 上加中间件与静态路由，
    # 而 FastAPI 的中间件必须在应用开始处理请求之前装好。
    # Mounted last: run_with installs middleware, which must precede any request.
    if gui:
        from agentic_tts.gui.app import mount as mount_gui

        mount_gui(app, module, path=gui_path)
        _logger.info("图形界面已挂在 %s（与服务共用一份权重）", gui_path)

    return app


__all__ = ["BatchBody", "HandoffBody", "SynthBody", "create_app"]
