"""
HTTP 服务 / HTTP service —— 端点与 `TTSModule` 的方法一一对应。

    GET  /health /info /engines /capabilities /voices
    POST /voices            新增/更新音色档案（含上传参考音频）
    POST /tts/synthesize    单段合成（可直接回 wav 二进制或 base64）
    POST /tts/design /tts/clone   语法糖，内部固定 mode
    POST /tts/batch         多段合成，每段落盘 + 可选拼接
    POST /tts/split         一整段文本 → 多段
    GET  /outputs/{run}/{file}    取产物

⚠️ **服务内只驻留一个 `TTSModule`，且合成是串行的**（一把锁）。
不是偷懒：8 GB 卡上并发两条请求会同时要两份权重，必然 OOM，而 OOM 的报错
完全看不出是并发导致的。要并发就横向起多个进程/节点，别在单进程里放。
One module, one lock: concurrent requests would need two copies of the weights.
"""

from __future__ import annotations

import base64
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field

from agentic_tts.audio import io as audio_io
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


def create_app(config: Config | str | None = None) -> FastAPI:
    """建 FastAPI 应用 / Build the FastAPI app（权重仍是懒加载的）。"""
    from agentic_tts.engine import TTSModule

    module = TTSModule(config)
    lock = threading.Lock()          # 见模块文档：单进程内串行合成

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        module.release()             # 退出时卸权重、清显存（仅 del 不还显存）

    app = FastAPI(title="Agentic TTS", version="0.1.0", lifespan=lifespan)

    # ------------------------------------------------------------ 查询 / info

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "backend": module.config.engine.backend}

    @app.get("/info")
    def info() -> dict:
        try:
            return module.info()
        except TTSError as exc:
            raise _http_error(exc) from exc

    @app.get("/engines")
    def engines() -> dict:
        return {"engines": module.engines()}

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

    return app


__all__ = ["BatchBody", "SynthBody", "create_app"]
