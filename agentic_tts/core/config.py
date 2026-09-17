"""
配置 / Configuration —— 一个 YAML 是唯一真源，少数几个环境变量可覆盖。

────────────────────────────────────────────────────────────────────────────
两条从别的项目吃过亏的规则
────────────────────────────────────────────────────────────────────────────
1. **相互依赖的路径统一相对一个根解析。** checkpoint 一律配「目录名」，相对
   `engine.models_dir`；改一个 `models_dir` 就整体搬家，不会出现「搬了一半、
   一半指向空气还不报错」（DailyNewsAssistant issue 006-A 的形状）。
   Every checkpoint resolves against `models_dir`, so relocating is all-or-nothing.
2. **`repo_dir` 默认指向本仓库内的 vendored 源码**，不引用别的项目目录。
   环境里 `qwen-tts` 的 editable 安装是坏的（finder 指向已删除目录），且磁盘上
   存在多份 commit 不同的源码 —— 只要靠 cwd 或全局安装碰运气，就会出现
   「同一条命令在不同目录导入不同源码」且毫无提示。
   The vendored copy under `third_party/` is the only source that gets imported.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from agentic_tts.core.errors import ConfigError
from agentic_tts.core.logging import get_logger

_logger = get_logger("config")

# 本文件位于 agentic_tts/core/config.py，上溯两层即项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"

_DTYPES = ("auto", "bfloat16", "float16", "float32")
_ATTN_IMPLS = ("eager", "sdpa", "flash_attention_2")

# API 后端供应商白名单 / API provider whitelist
_API_PROVIDERS = ("minimax", "dashscope", "openai", "elevenlabs")


def _str_to_bool(raw: str) -> bool:
    """环境变量字符串转 bool / String to bool for env vars."""
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "on", "y", "t"):
        return True
    if s in ("0", "false", "no", "off", "n", "f", ""):
        return False
    raise ValueError(f"无法解析为 bool：{raw!r}")


class EngineConfig(BaseModel):
    """引擎与设备 / Engine and device."""

    backend: str = "qwen3"
    device: str = "auto"                  # auto | cpu | cuda | cuda:0
    dtype: str = "auto"                   # auto → cpu:float32 / cuda:bfloat16
    attn_impl: str = "eager"
    seed: int = 20260907
    # 同时驻留几个 checkpoint。**8 GB 的 4060 只装得下一个 1.7B**（bf16 权重 3.9 GB、
    # 跑起来约 5 GB），所以默认 1：换 checkpoint 时先卸再载（轮动加载）。
    max_resident: int = 1
    vram_guard: bool = True               # 加载前预检显存，不够就带修改建议报错
    models_dir: str = ""
    repo_dir: str = ""                    # 空 = 用 third_party/qwen3_tts（vendored）
    checkpoints: dict[str, str] = Field(default_factory=lambda: {
        "custom_voice": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "voice_design": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "base": "Qwen3-TTS-12Hz-1.7B-Base",
    })
    options: dict[str, Any] = Field(default_factory=dict)   # 引擎私有参数


class TextConfig(BaseModel):
    """
    文本链路 / Text pipeline. 每一项都能关掉，便于 A/B。

    实测收益不均匀（11 条 A/B）：URL 不处理 CER 4.43→0、Markdown 0.83→0.17，
    而小数/emoji 处理了白做。`lexicon_*` 会**换掉字**，默认关闭。
    """

    prepare: bool = True
    markdown: bool = True
    urls: bool = True
    emoji: bool = True
    repeated_punct: bool = True
    numbers: bool = True
    abbreviations: bool = True
    lexicon_polyphones: bool = False      # 多音字替换：会换字，词表未逐条实测
    lexicon_proper_nouns: bool = False
    segment: bool = True
    max_chars: int = 60
    min_chars: int = 8
    events: bool = True                   # 解析 [pause:400ms] / [laugh] 之类标记


class AudioConfig(BaseModel):
    """音频与节奏 / Audio and rhythm."""

    sample_rate: Optional[int] = None     # None = 用引擎原生采样率
    format: str = "wav"
    pause_ms: int = 300                   # 同一角色的段间静音
    role_change_pause_ms: int = 600       # 换角色的段间静音（要能听出换人）
    event_pause_ms: int = 300             # `[pause]` 不带时长时的默认值
    peak_normalize: bool = False          # 拼接后是否做峰值归一


class GuardConfig(BaseModel):
    """
    失控兜底 / Runaway guard.

    实测：模型会停不下来，一直生成到 `max_new_tokens`，产出几十秒噪声或静音
    **且不报错**；静音检测与波形指标都测不出来（有一条 41 秒噪声，ASR 只回读出「嗯 k」）。
    主判据只能是「时长撞上限」。
    """

    enabled: bool = True
    retry: int = 1                        # 换种子重试次数
    cap_ratio: float = 0.95               # 时长 ≥ 上限 × 此值 ⇒ 判失控
    chars_per_sec: float = 5.0            # 中文约 5 字/秒（辅助判据）
    duration_ratio: float = 3.0           # 比「字数/5」长这么多倍也算可疑


class VoicesConfig(BaseModel):
    """音色档案 / Voice profiles."""

    dir: str = "./voices"
    file: str = "voices.yaml"
    # 角色 → 音色名。按角色拆分的多段文本靠这张表配音；GUI 也读它。
    roles: dict[str, str] = Field(default_factory=dict)
    default: Optional[str] = None         # 未指定音色时用哪个（None = 引擎默认）


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8300


class APIConfig(BaseModel):
    """API 后端配置 / API backend config."""

    enabled: bool = False
    provider: str = "minimax"           # minimax | dashscope | openai | elevenlabs
    endpoint: str = ""                  # 空 = 用 provider 默认
    api_key: str = ""                   # 空 = 必须从环境变量读（MINIMAX_API_KEY 等）
    group_id: str = ""                  # MiniMax 某些场景需要
    timeout_s: int = 60
    max_retries: int = 2
    retry_backoff_s: float = 1.0
    model: str = "speech-2.8-turbo"     # MiniMax 默认；不同 provider 自解析
    voice_cache_path: str = "./voices/api_voice_cache.json"
    rate_limit_per_min: int = 0         # 0 = 不限
    provider_options: dict[str, Any] = Field(default_factory=dict)

    # 自动 fallback 配置 / Auto fallback when local engine fails
    fallback_enabled: bool = True
    fallback_on_errors: list[str] = Field(default_factory=lambda: ["engine_error", "synth_error"])
    fallback_skip_voices: list[str] = Field(default_factory=list)


class GuiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8301


class OutputConfig(BaseModel):
    dir: str = "./outputs"
    # 字幕导出：每段音频一条字幕，时间轴按拼接后的整条算（含段间静音）。
    # SRT 剪映/Premiere/DaVinci 都能直接导入，所以不必退化成 txt。
    subtitles: bool = True
    subtitle_format: str = "srt"      # 目前只支持 srt


class Config(BaseModel):
    """
    全局配置 / The whole configuration.

    用法 / Usage::

        cfg = Config.load()                      # configs/config.yaml
        cfg = Config.load("path/to/other.yaml")
        cfg = Config()                           # 全默认（离线自测够用）
    """

    engine: EngineConfig = Field(default_factory=EngineConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    guard: GuardConfig = Field(default_factory=GuardConfig)
    voices: VoicesConfig = Field(default_factory=VoicesConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    gui: GuiConfig = Field(default_factory=GuiConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    # -- 加载 / loading ------------------------------------------------------

    @classmethod
    def load(cls, source: "Config | str | Path | None" = None) -> "Config":
        """
        读配置 / Load the configuration.

        参数 / Args:
            source: `Config` 原样返回；路径读该文件；None 读
                `configs/config.yaml`（不存在则全用默认值）。

        返回 / Returns:
            校验过的 `Config`（环境变量已覆盖）。

        抛出 / Raises:
            ConfigError: YAML 语法错、字段不合法，或显式指定的文件不存在。
        """
        if isinstance(source, Config):
            return source

        path = Path(source) if source else DEFAULT_CONFIG
        data: dict[str, Any] = {}
        if path.is_file():
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                raise ConfigError(f"配置文件 {path} 解析失败：{exc}") from exc
        elif source is not None:
            raise ConfigError(f"配置文件不存在：{path}")

        try:
            cfg = cls(**data)
        except Exception as exc:  # pydantic ValidationError
            raise ConfigError(f"配置不合法（{path}）：{exc}") from exc

        cfg._apply_env()
        cfg.validate_runtime()
        return cfg

    def _apply_env(self) -> None:
        """
        环境变量覆盖 / Environment overrides.

        只给最常在部署时改的那几项 —— 部署时不该为了换设备去改文件。
        Deliberately a short whitelist: deployment should not need to edit the file.
        """
        env = os.environ
        mapping = {
            "TTS_BACKEND": (self.engine, "backend", str),
            "TTS_DEVICE": (self.engine, "device", str),
            "TTS_DTYPE": (self.engine, "dtype", str),
            "TTS_SEED": (self.engine, "seed", int),
            "TTS_MAX_RESIDENT": (self.engine, "max_resident", int),
            "TTS_MODELS_DIR": (self.engine, "models_dir", str),
            "TTS_REPO_DIR": (self.engine, "repo_dir", str),
            "TTS_VOICES_DIR": (self.voices, "dir", str),
            "TTS_OUTPUT_DIR": (self.output, "dir", str),
            "TTS_SERVER_HOST": (self.server, "host", str),
            "TTS_SERVER_PORT": (self.server, "port", int),
            # API 后端环境变量
            "TTS_API_ENABLED": (self.api, "enabled", _str_to_bool),
            "TTS_API_PROVIDER": (self.api, "provider", str),
            "TTS_API_KEY": (self.api, "api_key", str),
            "TTS_API_GROUP_ID": (self.api, "group_id", str),
            "TTS_API_ENDPOINT": (self.api, "endpoint", str),
            "TTS_API_TIMEOUT_S": (self.api, "timeout_s", int),
            "TTS_API_MAX_RETRIES": (self.api, "max_retries", int),
            "TTS_API_MODEL": (self.api, "model", str),
            "TTS_API_FALLBACK": (self.api, "fallback_enabled", _str_to_bool),
        }
        for key, (section, attr, cast) in mapping.items():
            raw = env.get(key)
            if raw is None or raw == "":
                continue
            try:
                setattr(section, attr, cast(raw))
            except ValueError as exc:
                raise ConfigError(f"环境变量 {key}={raw!r} 不合法：{exc}") from exc

    def validate_runtime(self) -> None:
        """
        校验取值（不碰文件系统）/ Validate values without touching the filesystem.

        抛出 / Raises:
            ConfigError: dtype / attn 实现 / max_resident 不合法。
        """
        if self.engine.dtype not in _DTYPES:
            raise ConfigError(f"dtype={self.engine.dtype!r} 不合法　可选：{'、'.join(_DTYPES)}")
        if self.engine.attn_impl not in _ATTN_IMPLS:
            raise ConfigError(
                f"attn_impl={self.engine.attn_impl!r} 不合法　可选：{'、'.join(_ATTN_IMPLS)}")
        if self.engine.max_resident < 1:
            raise ConfigError("engine.max_resident 至少为 1")
        if self.text.max_chars <= self.text.min_chars:
            raise ConfigError(
                f"text.max_chars({self.text.max_chars}) 必须大于 "
                f"min_chars({self.text.min_chars})，否则分段会碎片化")
        # API 校验
        if self.api.enabled:
            if self.api.provider not in _API_PROVIDERS:
                raise ConfigError(
                    f"api.provider={self.api.provider!r} 不合法　可选：{'、'.join(_API_PROVIDERS)}")
            if self.api.timeout_s <= 0:
                raise ConfigError("api.timeout_s 必须为正")

    # -- 派生路径 / derived paths --------------------------------------------

    @staticmethod
    def _resolve(raw: str, default: Path) -> Path:
        """相对路径一律相对项目根解析 / Relative paths resolve against the project root."""
        if not raw:
            return default
        path = Path(raw)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @property
    def models_path(self) -> Path:
        return self._resolve(self.engine.models_dir, PROJECT_ROOT / "models")

    @property
    def repo_path(self) -> Path:
        """
        `qwen_tts` 源码所在目录 / Directory providing the `qwen_tts` package.

        默认是**本仓库内的 vendored 副本**，不指向任何外部项目。
        """
        return self._resolve(self.engine.repo_dir, PROJECT_ROOT / "third_party" / "qwen3_tts")

    @property
    def voices_path(self) -> Path:
        return self._resolve(self.voices.dir, PROJECT_ROOT / "voices")

    @property
    def voices_file(self) -> Path:
        return self.voices_path / self.voices.file

    @property
    def output_path(self) -> Path:
        return self._resolve(self.output.dir, PROJECT_ROOT / "outputs")

    def checkpoint_path(self, kind: str) -> Path:
        """
        取一类 checkpoint 的目录 / Resolve one checkpoint directory.

        参数 / Args:
            kind: `custom_voice` | `voice_design` | `base`

        返回 / Returns:
            绝对路径（**不保证存在**，存在性由 doctor / 加载时报错）。

        抛出 / Raises:
            ConfigError: kind 不在配置里。
        """
        raw = self.engine.checkpoints.get(kind)
        if not raw:
            raise ConfigError(
                f"配置里没有 checkpoint {kind!r}　已配：{'、'.join(self.engine.checkpoints) or '（无）'}")
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else self.models_path / raw


def ensure_qwen_tts_importable(repo_path: Path) -> Path:
    """
    让 `qwen_tts` 从**指定目录**可导入 / Make `qwen_tts` importable from one directory.

    **不修全局安装、不装包**：这个 conda 环境是多个项目共用的，动它会影响别人；
    而环境里那份 editable 安装本身就指向一个已删除的目录。这里只往 `sys.path`
    最前面插一条路径，并在之后校验**实际导入的是不是这一份**。
    Nothing global is mutated; the path is prepended and the resulting import verified.

    参数 / Args:
        repo_path: 含 `qwen_tts/` 子目录的目录 / a directory containing `qwen_tts/`.

    返回 / Returns:
        实际导入的 `qwen_tts` 包目录 / the package directory actually imported.

    抛出 / Raises:
        ConfigError: 目录不存在、导入失败，或导入到了**别的**副本。
    """
    package = repo_path / "qwen_tts"
    if not package.is_dir():
        raise ConfigError(
            f"{repo_path} 下没有 qwen_tts/ 子目录　"
            "把 configs/config.yaml 的 engine.repo_dir 指向 Qwen3-TTS 源码目录")

    entry = str(repo_path)
    if sys.path[:1] != [entry]:
        # 必须插到最前面：环境里存在同名的坏 editable 安装，也可能有别的副本
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)

    # 已经导入过别的副本就必须清掉，否则 sys.path 改了也没用
    imported = sys.modules.get("qwen_tts")
    if imported is not None:
        actual = Path(getattr(imported, "__file__", "") or "").resolve().parent
        if actual != package.resolve():
            for name in [n for n in sys.modules if n == "qwen_tts" or n.startswith("qwen_tts.")]:
                del sys.modules[name]

    try:
        import qwen_tts
    except ImportError as exc:
        raise ConfigError(
            f"路径已加入 sys.path 但仍导入不到 qwen_tts：{exc}　"
            "多半是依赖缺失（transformers / librosa / soundfile / einops）") from exc

    actual = Path(qwen_tts.__file__ or "").resolve().parent
    if actual != package.resolve():
        raise ConfigError(
            f"导入到了别的 qwen_tts 副本：{actual}　期望：{package}　"
            "环境里可能有一份坏的 editable 安装（见 docs/00_research.md §4）")
    _logger.debug("qwen_tts 来自 %s", actual)
    return actual


__all__ = [
    "DEFAULT_CONFIG",
    "PROJECT_ROOT",
    "APIConfig",
    "AudioConfig",
    "Config",
    "EngineConfig",
    "GuardConfig",
    "GuiConfig",
    "OutputConfig",
    "ServerConfig",
    "TextConfig",
    "VoicesConfig",
    "ensure_qwen_tts_importable",
]
