"""
交接单的测试 / Hand-off record tests —— 不加载权重，不起 GUI。

复测 / Re-run:
    python -m pytest tests/test_handoff.py -q

重点 / The point:
  · token 会**出现在 URL 里**并被用来拼路径 —— 非法 token 必须当作「没有」，
    否则 `../../` 就能读到输出目录之外的文件。
  · 交接单是**跨进程**契约（HTTP 服务写、GUI 读回、调用方再轮询），
    所以 `status` 与 `files` 的形状必须钉住：调用方靠 `files` 直接拼
    `/outputs/{run}/{file}` 来取产物。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentic_tts.core import handoff
from agentic_tts.server.app import create_app


@pytest.fixture()
def client(config) -> TestClient:
    return TestClient(create_app(config))


def test_create_then_read_round_trips(config) -> None:
    token = handoff.create(config.output_path, {"segments": ["第一段。", "第二段。"],
                                                "source": "DailyNewsAssistant"})
    record = handoff.read(config.output_path, token)
    assert record["segments"] == ["第一段。", "第二段。"]
    assert record["status"] == "pending"          # 新建即待处理
    assert record["source"] == "DailyNewsAssistant"


def test_unknown_token_is_none_not_an_error(config) -> None:
    """调用方轮询时会拿着过期 token 来问 —— 这不是异常情况。"""
    assert handoff.read(config.output_path, "20260101-000000-abcdef") is None


def test_path_traversal_token_is_rejected(config) -> None:
    """token 直接参与拼路径，`..` 必须在读之前就被挡住。"""
    assert handoff.read(config.output_path, "../../secret") is None
    assert handoff.update(config.output_path, "../../secret", status="done") is None


def test_update_marks_done_with_the_file_list(config) -> None:
    token = handoff.create(config.output_path, {"segments": ["一句话。"]})
    handoff.update(config.output_path, token, status="done",
                   files=["gui-1/seg_001.wav", "gui-1/merged.wav"], run="gui-1")
    record = handoff.read(config.output_path, token)
    assert record["status"] == "done"
    # 调用方就是拿这两个相对路径去 GET /outputs/{run}/{file}
    assert record["files"] == ["gui-1/seg_001.wav", "gui-1/merged.wav"]


def test_http_handoff_returns_a_gui_url(client, config) -> None:
    body = client.post("/gui/handoff", json={"segments": ["今天的资讯。"],
                                             "source": "DNA"}).json()
    assert body["token"]
    assert body["gui_url"].endswith(f"?import={body['token']}")
    assert str(config.gui.port) in body["gui_url"]


def test_http_handoff_accepts_plain_text(client) -> None:
    """只给 text 时当成一段 —— 单段调用方不必先自己拆条。"""
    token = client.post("/gui/handoff", json={"text": "一整段话。"}).json()["token"]
    assert client.get(f"/gui/handoff/{token}").json()["segments"] == ["一整段话。"]


def test_http_handoff_rejects_empty_text(client) -> None:
    assert client.post("/gui/handoff", json={"segments": ["  "]}).status_code == 400


def test_http_read_unknown_handoff_is_404(client) -> None:
    assert client.get("/gui/handoff/20260101-000000-abcdef").status_code == 404
