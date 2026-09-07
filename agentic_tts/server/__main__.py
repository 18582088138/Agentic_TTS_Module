"""
起服务 / Start the service::

    python -m agentic_tts.server                       # 读 configs/config.yaml
    python -m agentic_tts.server --engine sine         # 离线自测（不加载权重）
    python -m agentic_tts.server --port 8300
"""

from __future__ import annotations

import argparse

import uvicorn

from agentic_tts.core.config import Config
from agentic_tts.server.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic TTS HTTP 服务")
    parser.add_argument("--config", "-c", default=None)
    parser.add_argument("--engine", "-e", default=None, help="临时覆盖引擎：qwen3 | sine")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    config = Config.load(args.config)
    if args.engine:
        config.engine.backend = args.engine
    uvicorn.run(create_app(config),
                host=args.host or config.server.host,
                port=args.port or config.server.port)


if __name__ == "__main__":
    main()
