"""
交接单 / Hand-off records —— 把「要念的文本」从别的应用交到 TTS GUI 手上。

场景：DailyNewsAssistant 想让人在 TTS 的多段界面里精修（换音色、克隆、逐句重掷），
于是把稿子 POST 过来拿一个 token，再打开 `GUI?import=<token>`；
生成完 GUI 把产物清单写回同一条记录，调用方轮询到 `status=done` 就来取文件。
A caller posts a script, gets a token, opens the GUI with it, and later polls the same
record for the produced files.

为什么落成文件而不是放内存 / Why a file rather than an in-process dict:
    HTTP 服务与 GUI 是**两个进程**（`cli serve` 与 `cli gui` 各起一个），
    内存里的字典彼此看不见。写在 `outputs/handoff/` 下两边都读得到，重启也不丢。
    The service and the GUI are separate processes; an in-memory dict is invisible to
    the other one.

为什么由服务端落盘而不是调用方直接写文件 / Why the service writes it:
    调用方可能在另一台机器上，写不了这台机器的磁盘。它只发 HTTP。
    The caller may live on another machine and can only speak HTTP.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

# token 会出现在 URL 里，因此**必须限定字符集**：
# 直接拿它拼路径而不校验，`../../` 就能读写输出目录之外的文件。
# The token arrives in a URL and is used to build a path, so the character set is fixed;
# without this check `../../` would escape the directory.
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")

# 过期时间：交接单只在「人正在界面上操作」这段时间里有用
# Records only matter while someone is working in the GUI.
_TTL_SECONDS = 7 * 24 * 3600


def handoff_dir(root: Path) -> Path:
    """交接单目录 / Where records live（`<outputs>/handoff/`）。"""
    return Path(root) / "handoff"


def create(root: Path, payload: dict[str, Any]) -> str:
    """
    新建一张交接单 / Create one record, returning its token.

    token 带时间前缀只是为了肉眼可排序；唯一性靠后面那段随机。
    """
    directory = handoff_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    _prune(directory)

    token = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    record = {
        "token": token,
        "created_at": time.time(),
        "status": "pending",
        "segments": [],
        "files": [],
        "run": "",
        **payload,
    }
    _path(directory, token).write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return token


def read(root: Path, token: str) -> Optional[dict[str, Any]]:
    """读一张交接单 / Read one record；token 非法或不存在都回 None。"""
    if not _TOKEN_RE.match(token or ""):
        return None
    path = _path(handoff_dir(root), token)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def update(root: Path, token: str, **fields: Any) -> Optional[dict[str, Any]]:
    """
    更新一张交接单 / Merge fields into one record.

    GUI 生成完就是靠这个把 `status=done` 与产物清单写回去的。
    """
    record = read(root, token)
    if record is None:
        return None
    record.update(fields)
    record["updated_at"] = time.time()
    _path(handoff_dir(root), token).write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return record


# ---------------------------------------------------------------- 内部 / internals


def _path(directory: Path, token: str) -> Path:
    return directory / f"{token}.json"


def _prune(directory: Path) -> None:
    """删掉过期的交接单 / Drop expired records（顺手做，不值得起后台任务）。"""
    deadline = time.time() - _TTL_SECONDS
    for path in directory.glob("*.json"):
        try:
            if path.stat().st_mtime < deadline:
                path.unlink()
        except OSError:      # 正被别的进程读/删，跳过就好
            continue


__all__ = ["create", "handoff_dir", "read", "update"]
