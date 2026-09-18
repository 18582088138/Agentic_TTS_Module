"""
音色档案 / Voice profiles —— 把「内置音色」「音色设计」「参考音频克隆」统一成一个名字。

调用方只写 `voice="anchor_f"`，不必知道背后是 `generate_custom_voice` 还是
`generate_voice_clone`（那是两个不同的 checkpoint）。这层的价值就在这里：
**把"用哪套 API、载哪份权重"从调用方手里拿走。**
Callers name a voice; which API and which checkpoint it needs is not their problem.

档案文件 `voices/voices.yaml`::

    serena:
      mode: custom_voice
      speaker: Serena
      language: Chinese
    anchor_f:
      mode: voice_clone
      ref_audio: refs/anchor_f.wav      # 相对 voices/ 解析
      ref_text: "今天的人工智能资讯"
      x_vector_only: false
    calm_f:
      mode: voice_design
      instruct: "沉稳的女声，语速平缓，播音腔"

⚠️ `ref_audio` 相对 `voices/` 解析 —— **所有相互依赖的路径统一相对一个根**，
否则改一个目录就会出现「档案搬了、参考音频留在原地，且指向空气不报错」。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from agentic_tts.core.config import Config
from agentic_tts.core.errors import ConfigError
from agentic_tts.core.logging import get_logger
from agentic_tts.core.types import Mode, SynthRequest, VoiceInfo

_logger = get_logger("voices")


@dataclass
class ResolvedVoice:
    """
    档案 + 请求合并后的最终取值 / The effective values after merging profile and request.

    合并规则：**请求里显式给了的字段优先**，其余用档案填。
    这样「用 anchor_f 这个音色，但这一段换个 instruct」是自然可写的。
    """

    mode: Mode
    voice: Optional[str] = None
    speaker: Optional[str] = None
    language: str = "Chinese"
    instruct: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    x_vector_only: bool = False


class VoiceStore:
    """音色档案表 / The voice profile table（内置音色 + 本地档案）。"""

    def __init__(self, config: Config, builtin: Optional[list[VoiceInfo]] = None) -> None:
        self.config = config
        self._builtin = {v.name: v for v in (builtin or [])}
        self._builtin_lower = {k.lower(): k for k in self._builtin}
        self._profiles: dict[str, VoiceInfo] = {}
        self.reload()

    # ------------------------------------------------------------------ 读写

    def reload(self) -> None:
        """
        重读档案文件 / Re-read the profile file.

        文件不存在**不是错误** —— 只用内置音色也能工作，第一次保存时再建。
        """
        path = self.config.voices_file
        self._profiles = {}
        if not path.is_file():
            return
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"音色档案 {path} 解析失败：{exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"音色档案 {path} 的顶层应是「名字: 配置」的映射")

        for name, raw in data.items():
            if not isinstance(raw, dict):
                raise ConfigError(f"音色档案 {name!r} 的内容应是映射，实际是 {type(raw).__name__}")
            try:
                self._profiles[str(name)] = VoiceInfo(
                    name=str(name),
                    mode=Mode(raw.get("mode", "custom_voice")),
                    engine=str(raw.get("engine", self.config.engine.backend)),
                    language=raw.get("language"),
                    description=raw.get("description"),
                    builtin=False,
                    speaker=raw.get("speaker"),
                    instruct=raw.get("instruct"),
                    ref_audio=raw.get("ref_audio"),
                    ref_text=raw.get("ref_text"),
                    x_vector_only=bool(raw.get("x_vector_only", False)),
                )
            except ValueError as exc:
                raise ConfigError(f"音色档案 {name!r} 不合法：{exc}") from exc
        _logger.debug("载入 %d 个音色档案", len(self._profiles))

    def save(self, info: VoiceInfo) -> Path:
        """
        新增/更新一个档案并落盘 / Upsert one profile and persist it.

        GUI 里「把这段参考音频存成一个音色」走的就是这里。
        """
        path = self.config.voices_file
        path.parent.mkdir(parents=True, exist_ok=True)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
        data = data or {}
        entry: dict[str, Any] = {"mode": info.mode.value}
        for key in ("engine", "language", "description", "speaker", "instruct",
                    "ref_audio", "ref_text"):
            value = getattr(info, key)
            if value:
                entry[key] = value
        if info.mode is Mode.VOICE_CLONE:
            entry["x_vector_only"] = info.x_vector_only
        data[info.name] = entry
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=True),
                        encoding="utf-8")
        self._profiles[info.name] = info.model_copy(update={"builtin": False})
        return path

    def delete(self, name: str) -> bool:
        """删掉一个档案 / Remove one profile（内置音色删不掉）。"""
        path = self.config.voices_file
        if not path.is_file():
            return False
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if name not in data:
            return False
        data.pop(name)
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=True),
                        encoding="utf-8")
        self._profiles.pop(name, None)
        return True

    # ------------------------------------------------------------------ 查询

    def all(self) -> list[VoiceInfo]:
        """全部可选音色 / Every selectable voice（档案在前，内置在后）。"""
        return [*self._profiles.values(), *self._builtin.values()]

    def get(self, name: str) -> Optional[VoiceInfo]:
        """
        按名字取音色 / Look up one voice.

        档案优先于内置同名项 —— 用户显式写下的档案应该能覆盖内置默认。
        内置音色名**大小写不敏感**（官方是 `Serena`，但没人愿意背大小写）。
        """
        if name in self._profiles:
            return self._profiles[name]
        if name in self._builtin:
            return self._builtin[name]
        actual = self._builtin_lower.get((name or "").lower())
        return self._builtin[actual] if actual else None

    def role_voice(self, role: Optional[str]) -> Optional[str]:
        """角色 → 音色名 / Map a role to a voice name（`voices.roles` 那张表）。"""
        if not role:
            return None
        return self.config.voices.roles.get(role)

    def resolve_ref_audio(self, ref_audio: Optional[str]) -> Optional[str]:
        """
        参考音频路径解析 / Resolve a reference-audio path.

        相对路径一律相对 `voices/` —— 见模块文档里「统一相对一个根」那一条。
        """
        if not ref_audio or ref_audio.startswith(("data:audio", "http://", "https://")):
            return ref_audio
        path = Path(ref_audio)
        if not path.is_absolute():
            path = self.config.voices_path / path
        return str(path)

    # ------------------------------------------------------------------ 合并

    def resolve(self, req: SynthRequest) -> ResolvedVoice:
        """
        把档案套到请求上 / Merge a profile into a request.

        参数 / Args:
            req: 请求；`voice` 可以是内置音色名、档案名，或空。

        返回 / Returns:
            `ResolvedVoice` —— 引擎需要的全部字段都已定好。

        抛出 / Raises:
            ConfigError: 指定了一个不存在的音色名。**不静默回落到默认音色** ——
                拼错名字却听到默认音色，是最难查的一类问题。
        """
        name = req.voice
        profile: Optional[VoiceInfo] = None
        if name:
            profile = self.get(name)
            if profile is None:
                available = "、".join(sorted(v.name for v in self.all())) or "（无）"
                raise ConfigError(f"没有名为 {name!r} 的音色　可选：{available}")
        elif self.config.voices.default:
            profile = self.get(self.config.voices.default)

        # 模式：请求显式指定 > 档案 > 从参数推断
        mode = req.mode or (profile.mode if profile else None) or req.resolved_mode()

        speaker = profile.speaker if profile and profile.mode is Mode.CUSTOM_VOICE else None
        if mode is Mode.CUSTOM_VOICE and not speaker:
            # 档案没给 speaker（例如直接写了内置音色名）就用音色名本身
            speaker = (profile.name if profile else None) or name

        instruct = req.instruct or (profile.instruct if profile else None)
        ref_audio = req.ref_audio or (profile.ref_audio if profile else None)
        ref_text = req.ref_text or (profile.ref_text if profile else None)
        x_vector_only = req.x_vector_only or bool(profile and profile.x_vector_only)
        language = req.language or (profile.language if profile else None) or "Chinese"

        if mode is Mode.VOICE_CLONE:
            ref_audio = self.resolve_ref_audio(ref_audio)
            if not ref_audio:
                raise ConfigError(f"音色 {name!r} 是克隆模式但没有 ref_audio")
            # 注：MiniMax /v1/voice_clone 接口允许 ref_text 为空（只用 speaker embedding），
            # 所以这里不再硬要求 ref_text（原先是给 Qwen3-TTS 的 ICL 模式强校验）
        if mode is Mode.VOICE_DESIGN and not instruct:
            raise ConfigError(f"音色 {name!r} 是音色设计模式但没有 instruct 描述")

        return ResolvedVoice(
            mode=mode,
            voice=name or (profile.name if profile else None),
            speaker=speaker,
            language=language,
            instruct=instruct,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only=x_vector_only,
        )


__all__ = ["ResolvedVoice", "VoiceStore"]
