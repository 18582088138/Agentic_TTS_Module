"""
真 API demo 后端 / Real API demo backend.

启动：python server.py
访问：http://0.0.0.0:8310/

调真 MiniMax（用 Mavis secret 里的 key）；沙箱启动后从 .mavis_secret 读 key。
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Agentic TTS API Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# MiniMax 配置
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY") or os.environ.get("TTS_API_KEY")
MINIMAX_ENDPOINT = os.environ.get("MINIMAX_ENDPOINT", "https://api.minimax.chat")
MINIMAX_MODEL = os.environ.get("MINIMAX_MODEL", "speech-2.8-turbo")

CACHE_PATH = Path(os.environ.get("VOICE_CACHE_PATH", "/tmp/api_voice_cache.json"))
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

# 简易 voice_id 缓存
_cache: dict[str, dict[str, Any]] = {}
if CACHE_PATH.is_file():
    try:
        _cache = json.loads(CACHE_PATH.read_text())
    except Exception:
        _cache = {}


def cache_key(audio_bytes: bytes, ref_text: str | None) -> str:
    digest = hashlib.sha256(audio_bytes).hexdigest()
    return f"{digest}|{ref_text or ''}"


class SynthReq(BaseModel):
    text: str
    mode: str = "custom_voice"  # custom_voice | voice_design | voice_clone
    speaker: str | None = None  # custom_voice 时用
    instruct: str | None = None  # voice_design 时用
    ref_audio_base64: str | None = None  # voice_clone 时用
    ref_text: str | None = None
    sample_rate: int = 24000


def normalize_events(text: str) -> tuple[str, list[str]]:
    """[laugh] -> (laughs) 等 MiniMax 2.8 原生映射"""
    import re
    warnings: list[str] = []
    tag_map = {
        "laugh": "(laughs)", "sigh": "(sighs)", "breath": "(breath)",
        "cry": "(crying)", "cough": "(coughs)", "sniff": "(sniffs)",
        "humming": "(humming)", "whistle": "(whistles)",
    }
    for tag, native in tag_map.items():
        if re.search(rf"\[{tag}\]", text):
            text = re.sub(rf"\[{tag}\]", native, text)
            warnings.append(f"[{tag}] -> {native}")
    text = re.sub(r"\[pause:(\d+)ms\]", lambda m: f"<#{int(m.group(1))/1000:.2f}#>", text)
    text = re.sub(r"\[pause\]", "<#0.3#>", text)
    return text, warnings


async def upload_audio(client: httpx.AsyncClient, audio_bytes: bytes) -> int:
    """POST /v1/files/upload"""
    headers = {"Authorization": f"Bearer {MINIMAX_API_KEY}"}
    files = {"file": ("ref.mp3", audio_bytes, "audio/mpeg")}
    data = {"purpose": "voice_clone"}
    r = await client.post(f"{MINIMAX_ENDPOINT}/v1/files/upload",
                          headers=headers, files=files, data=data, timeout=60)
    r.raise_for_status()
    return int(r.json()["file"]["file_id"])


async def clone_voice(client: httpx.AsyncClient, file_id: int, voice_id: str,
                      text: str, model: str) -> str:
    """POST /v1/voice_clone"""
    headers = {"Authorization": f"Bearer {MINIMAX_API_KEY}",
               "Content-Type": "application/json"}
    body = {
        "file_id": file_id, "voice_id": voice_id, "text": text, "model": model,
        "language_boost": "auto", "need_noise_reduction": False,
        "need_volume_normalization": False,
    }
    r = await client.post(f"{MINIMAX_ENDPOINT}/v1/voice_clone",
                          headers=headers, json=body, timeout=60)
    r.raise_for_status()
    return r.json().get("voice_id", voice_id)


async def t2a(client: httpx.AsyncClient, body: dict) -> bytes:
    """POST /v1/t2a_v2 → 返回 wav 字节"""
    headers = {"Authorization": f"Bearer {MINIMAX_API_KEY}",
               "Content-Type": "application/json"}
    r = await client.post(f"{MINIMAX_ENDPOINT}/v1/t2a_v2",
                          headers=headers, json=body, timeout=60)
    r.raise_for_status()
    data = r.json()
    if (data.get("base_resp") or {}).get("status_code", 0) != 0:
        raise HTTPException(400, f"MiniMax 业务错：{data['base_resp']}")
    return bytes.fromhex(data["audio_file"])


@app.get("/api/info")
def info() -> dict:
    return {
        "key_set": bool(MINIMAX_API_KEY),
        "key_masked": (MINIMAX_API_KEY[:6] + "***" + MINIMAX_API_KEY[-4:]
                       if MINIMAX_API_KEY else ""),
        "endpoint": MINIMAX_ENDPOINT,
        "model": MINIMAX_MODEL,
        "voice_cache_size": len(_cache),
        "voice_cache_path": str(CACHE_PATH),
    }


@app.post("/api/synthesize")
async def synthesize(req: SynthReq) -> dict:
    if not MINIMAX_API_KEY:
        raise HTTPException(503, "MINIMAX_API_KEY 未配置")

    text, warnings = normalize_events(req.text)

    voice_id = req.speaker or "male-qn-qingse"
    cache_hit = False

    async with httpx.AsyncClient() as client:
        # voice_clone 流程
        if req.mode == "voice_clone":
            if not req.ref_audio_base64:
                raise HTTPException(400, "voice_clone 需要 ref_audio_base64")
            audio_bytes = base64.b64decode(req.ref_audio_base64)
            key = cache_key(audio_bytes, req.ref_text)
            cached = _cache.get(key)
            if cached:
                voice_id = cached["voice_id"]
                cache_hit = True
            else:
                file_id = await upload_audio(client, audio_bytes)
                voice_id = f"v_{key[:8]}_{int(time.time()) % 10**6:06d}"
                voice_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in voice_id)[:64]
                voice_id = await clone_voice(client, file_id, voice_id,
                                            req.ref_text or "这是用于试听的声音克隆样本。",
                                            MINIMAX_MODEL)
                _cache[key] = {"voice_id": voice_id, "provider": "minimax",
                               "ref_text": req.ref_text, "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                CACHE_PATH.write_text(json.dumps(_cache, indent=2))

        body = {
            "model": MINIMAX_MODEL,
            "text": text,
            "stream": False,
            "language_boost": "auto",
            "output_format": "hex",
            "voice_setting": {"voice_id": voice_id, "speed": 1.0, "vol": 1.0, "pitch": 0},
            "audio_setting": {"sample_rate": req.sample_rate, "bitrate": 128000,
                              "format": "wav", "channel": 1},
        }
        if req.instruct:
            body["voice_setting"]["instruct"] = req.instruct

        try:
            wav = await t2a(client, body)
        except httpx.HTTPStatusError as e:
            raise HTTPException(502, f"MiniMax {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            raise HTTPException(502, f"调用失败：{e}")

    return {
        "ok": True,
        "audio_base64": base64.b64encode(wav).decode(),
        "audio_size_bytes": len(wav),
        "sample_rate": req.sample_rate,
        "voice_id_used": voice_id,
        "cache_hit": cache_hit,
        "warnings": warnings,
        "model": MINIMAX_MODEL,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8310))
    uvicorn.run(app, host="0.0.0.0", port=port)