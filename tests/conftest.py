"""
测试公共夹具 / Shared test fixtures.

复测 / Re-run everything offline:
    conda activate ov_env_py312
    cd Agent_TTS_Module
    python -m pytest tests -q                 # 只跑离线用例（秒级）
    python -m pytest tests -q -m real         # 真实模型（手动，每条几十秒起）

默认套件**不加载任何权重**：全部走 `sine` 引擎与纯函数。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 允许不安装就跑测试（开发机上 `pip install -e .` 不是前提）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_tts.core.config import Config  # noqa: E402


@pytest.fixture()
def config(tmp_path: Path) -> Config:
    """
    离线配置 / An offline configuration.

    产物与音色档案都落到 tmp_path —— 测试**不能**写进项目的 outputs/voices，
    否则跑一遍测试就污染真实产物目录。
    """
    cfg = Config()
    cfg.engine.backend = "sine"
    cfg.output.dir = str(tmp_path / "outputs")
    cfg.voices.dir = str(tmp_path / "voices")
    return cfg


@pytest.fixture()
def module(config: Config):
    """接好 sine 引擎的门面 / The facade wired to the sine engine."""
    from agentic_tts.engine import TTSModule

    return TTSModule(config)
