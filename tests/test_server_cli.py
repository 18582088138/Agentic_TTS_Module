"""
HTTP 服务与 CLI 的测试 / HTTP service and CLI tests —— 走 `sine`，不加载权重。

复测 / Re-run:
    python -m pytest tests/test_server_cli.py -q

重点 / The point:
  · 端点与门面方法一一对应，且**异常映射成有意义的状态码**（400 改参数 /
    409 换引擎 / 422 合成失败）—— 调用方要能自动处置，而不是看一个 500。
  · 产物下载必须挡住目录穿越（`..` 解析后再校验）。
  · CLI 只做参数解析与展示，业务全在门面 —— 所以这里只验"能跑通、能报错"。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from agentic_tts.cli import app as cli_app
from agentic_tts.server.app import create_app


@pytest.fixture()
def client(config) -> TestClient:
    return TestClient(create_app(config))


@pytest.fixture()
def config_file(config, tmp_path) -> str:
    """把夹具配置写成 YAML，供 CLI 用 `--config` 读 / A YAML file for the CLI."""
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True),
                    encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------- HTTP


def test_health_and_capabilities(client) -> None:
    assert client.get("/health").json()["status"] == "ok"
    body = client.get("/capabilities").json()
    assert "custom_voice" in body["capabilities"]


def test_engines_lists_placeholders_too(client) -> None:
    """GUI 靠这个置灰开关，所以未接入的引擎也要出现在表里。"""
    names = {e["name"] for e in client.get("/engines").json()["engines"]}
    assert {"qwen3", "sine", "breeze2", "index2"} <= names


def test_voices_endpoint(client) -> None:
    body = client.get("/voices").json()
    assert any(v["name"] == "test_female" for v in body["voices"])


def test_synthesize_returns_metadata(client) -> None:
    response = client.post("/tts/synthesize",
                           json={"text": "这是一句测试。", "voice": "test_female"})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] and body["seconds"] > 0
    assert Path(body["path"]).is_file()


def test_synthesize_can_return_wav_bytes(client) -> None:
    response = client.post("/tts/synthesize",
                           json={"text": "这是一句测试。", "voice": "test_female",
                                 "encoding": "wav", "save": False})
    assert response.headers["content-type"] == "audio/wav"
    assert response.content[:4] == b"RIFF"


def test_synthesize_base64(client) -> None:
    body = client.post("/tts/synthesize",
                       json={"text": "这是一句测试。", "voice": "test_female",
                             "encoding": "base64", "save": False}).json()
    assert body["audio_base64"]


def test_bad_request_is_422_from_pydantic(client) -> None:
    """ICL 克隆缺 ref_text —— 请求层就该拦住，不该走到加载权重。"""
    response = client.post("/tts/clone", json={"text": "x", "ref_audio": "a.wav"})
    assert response.status_code == 422


def test_unknown_voice_is_400(client) -> None:
    response = client.post("/tts/synthesize", json={"text": "x", "voice": "不存在的音色"})
    assert response.status_code == 400
    assert "可选" in response.json()["detail"]


def test_unrecoverable_runaway_is_422(client) -> None:
    response = client.post("/tts/synthesize",
                           json={"text": "这一句一直失控。", "voice": "test_male",
                                 "gen": {"_simulate_runaway": 9}})
    assert response.status_code == 422
    assert "撞上限" in response.json()["detail"]


def test_batch_endpoint(client) -> None:
    response = client.post("/tts/batch", json={
        "segments": [
            {"text": "第一段的内容。", "voice": "test_female", "role": "旁白"},
            {"text": "第二段的内容。", "voice": "test_male", "role": "记者"},
        ],
        "merge": True, "name": "http_batch"})
    body = response.json()
    assert body["ok"] and len(body["segments"]) == 2
    assert Path(body["path"]).name == "merged.wav"


def test_split_endpoint(client) -> None:
    body = client.post("/tts/split", json={
        "text": "旁白：第一句。\n记者：第二句。", "rule": "role",
        "role_voices": {"旁白": "test_female"}}).json()
    assert [s["role"] for s in body["segments"]] == ["旁白", "记者"]
    assert body["segments"][0]["voice"] == "test_female"


def test_voice_profile_roundtrip(client) -> None:
    saved = client.post("/voices", json={"name": "calm", "mode": "voice_design",
                                         "engine": "sine", "instruct": "沉稳女声"})
    assert saved.status_code == 200
    assert any(v["name"] == "calm" for v in client.get("/voices").json()["voices"])
    assert client.delete("/voices/calm").json()["deleted"] is True


def test_output_download_blocks_traversal(client) -> None:
    """`..` 必须在**解析后**被挡住；只比字符串是能绕过去的。"""
    assert client.get("/outputs/..%2f../secret.wav").status_code in (404, 400)


def test_output_download_serves_real_file(client) -> None:
    body = client.post("/tts/synthesize",
                       json={"text": "这是一句测试。", "voice": "test_female",
                             "name": "dl"}).json()
    filename = Path(body["path"]).name
    response = client.get(f"/outputs/dl/{filename}")
    assert response.status_code == 200
    assert response.content[:4] == b"RIFF"


# ----------------------------------------------------------------- CLI


def test_cli_engines_table(config_file) -> None:
    result = CliRunner().invoke(cli_app, ["engines", "--config", config_file])
    assert result.exit_code == 0
    assert "qwen3" in result.stdout and "未接入" in result.stdout


def test_cli_voices(config_file) -> None:
    result = CliRunner().invoke(cli_app, ["voices", "--config", config_file, "-e", "sine"])
    assert result.exit_code == 0
    assert "test_female" in result.stdout


def test_cli_speak(config_file) -> None:
    result = CliRunner().invoke(cli_app, [
        "speak", "这是一句命令行测试。", "-v", "test_female", "-e", "sine",
        "--config", config_file, "-n", "cli"])
    assert result.exit_code == 0
    # 音频时长与耗时必须**分开报**（只给一个数字会被读成"这次跑了多久"）
    assert "音频" in result.stdout and "耗时" in result.stdout


def test_cli_speak_reports_errors_as_one_line(config_file) -> None:
    result = CliRunner().invoke(cli_app, [
        "speak", "x", "-v", "不存在的音色", "-e", "sine", "--config", config_file])
    assert result.exit_code == 1
    assert "✗" in result.stdout


def test_cli_split_json(config_file) -> None:
    result = CliRunner().invoke(cli_app, [
        "split", "第一句话。第二句话。", "--rule", "punct", "--json", "--config", config_file])
    assert result.exit_code == 0
    assert "第一句话" in result.stdout


def test_cli_batch_from_yaml(config_file, tmp_path) -> None:
    segments = tmp_path / "segments.yaml"
    segments.write_text(
        "- text: 第一段内容。\n  voice: test_female\n- text: 第二段内容。\n  voice: test_male\n",
        encoding="utf-8")
    result = CliRunner().invoke(cli_app, [
        "batch", str(segments), "-e", "sine", "--config", config_file, "-n", "cli_batch"])
    assert result.exit_code == 0
    assert "共 2 段" in result.stdout


def test_cli_doctor_on_sine_backend(config_file) -> None:
    """sine 后端不该去查 checkpoint —— 体检必须能在没有权重的机器上跑完。"""
    result = CliRunner().invoke(cli_app, ["doctor", "--config", config_file, "-e", "sine"])
    assert result.exit_code == 0
    assert "体检通过" in result.stdout
