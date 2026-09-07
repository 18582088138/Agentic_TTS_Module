"""
最小 HTTP 客户端 / A minimal HTTP client —— 给下游项目（DNA / K12）抄的样子。

    python scripts/client.py --url http://127.0.0.1:8300 info
    python scripts/client.py speak "今天的人工智能资讯" --voice Serena -o out.wav
    python scripts/client.py batch segments.yaml -o merged.wav

只用 `requests`（没装就用 `urllib`），刻意不依赖本模块 ——
它要能被拷到任何一个调用方项目里直接用。
Deliberately standalone so it can be copied into any caller project.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any


def post(url: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:      # noqa: S310 - 本地服务
        return json.loads(response.read())


def get(url: str, path: str) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + path) as response:   # noqa: S310
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic TTS HTTP 客户端")
    parser.add_argument("--url", default="http://127.0.0.1:8300")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info")
    sub.add_parser("voices")

    speak = sub.add_parser("speak")
    speak.add_argument("text")
    speak.add_argument("--voice", "-v", default=None)
    speak.add_argument("--instruct", "-i", default=None)
    speak.add_argument("--out", "-o", default=None, help="把音频存到本地（走 base64 取回）")

    batch = sub.add_parser("batch")
    batch.add_argument("file", help="段落列表（YAML/JSON）")
    batch.add_argument("--out", "-o", default=None)

    args = parser.parse_args()

    if args.command == "info":
        print(json.dumps(get(args.url, "/info"), ensure_ascii=False, indent=2))
        return
    if args.command == "voices":
        for voice in get(args.url, "/voices")["voices"]:
            print(f"{voice['name']:<16}{voice['mode']:<14}{voice.get('description') or ''}")
        return

    if args.command == "speak":
        payload: dict[str, Any] = {"text": args.text, "voice": args.voice,
                                   "instruct": args.instruct,
                                   "encoding": "base64" if args.out else "json"}
        result = post(args.url, "/tts/synthesize", {k: v for k, v in payload.items()
                                                    if v is not None})
    else:
        import yaml

        segments = yaml.safe_load(Path(args.file).read_text(encoding="utf-8"))
        result = post(args.url, "/tts/batch",
                      {"segments": segments, "merge": True,
                       "encoding": "base64" if args.out else "json"})

    if args.out and result.get("audio_base64"):
        Path(args.out).write_bytes(base64.b64decode(result["audio_base64"]))
        print(f"已写入 {args.out}")
    print(f"{result['seconds']}s　RTF {result['rtf']}　服务端产物 {result.get('path')}")
    for warning in result.get("warnings", []):
        print(f"! {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
