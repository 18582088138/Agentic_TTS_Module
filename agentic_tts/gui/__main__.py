"""
起界面 / Start the GUI::

    python -m agentic_tts.gui                    # 读 configs/config.yaml
    python -m agentic_tts.gui --engine sine      # 离线试界面（不加载权重）
"""

from __future__ import annotations

import argparse

from agentic_tts.gui.app import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic TTS 图形界面")
    parser.add_argument("--config", "-c", default=None)
    parser.add_argument("--engine", "-e", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    run(config=args.config, engine=args.engine, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
