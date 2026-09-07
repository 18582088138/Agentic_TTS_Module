# 06 · 怎么接一个新 TTS 引擎 / Adding an engine

接一个引擎要写的东西只有两样：**一个 `synthesize()` + 一张能力表。**
文本预处理、分段、事件标记、失控兜底、拼接插静音、轮动加载排序全部在门面上，
自动继承 —— 这是刻意的分层，不要在引擎里重写它们（否则会出现 N 份互相偷偷
不一致的分段规则，长文案上给出不同切法，调试时看不出是哪份在起作用）。

---

## 1 · 四步

### ① 建文件 `agentic_tts/engines/mytts.py`

```python
from agentic_tts.core.registry import register_engine
from agentic_tts.core.types import Capability, Mode, RawAudio, VoiceInfo
from agentic_tts.engines.base import SynthChunk, TTSEngine


@register_engine("mytts")
class MyTTSEngine(TTSEngine):
    """一句话说明（会出现在 `tts engines` 表里）。"""

    name = "mytts"
    codes_per_second = 12.5        # 声学 token 帧率，用于把 max_new_tokens 换成时长上限
    default_max_new_tokens = 4096  # 没有的话失控判据只剩启发式那一条

    @classmethod
    def declared_capabilities(cls) -> set[Capability]:
        # **静态**声明：GUI 开机时据此置灰开关，不能要求先加载权重
        return {Capability.CUSTOM_VOICE, Capability.VOICE_CLONE, Capability.CLONE_ICL}

    def builtin_voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(name="alice", mode=Mode.CUSTOM_VOICE, engine=self.name,
                          language="Chinese", speaker="alice")]

    def languages(self) -> list[str]:
        return ["Chinese", "English"]

    def synthesize(self, chunk: SynthChunk) -> RawAudio:
        self.require_mode(chunk.mode)          # 不支持就报错，**绝不静默降级**
        model = self._load()                   # 懒加载；重依赖在函数体里 import
        wav, rate = model.tts(chunk.text, speaker=chunk.speaker, seed=chunk.seed)
        return RawAudio(wav=wav, sample_rate=rate,
                        effective_gen={...},   # 尽量填**实际生效**的参数，不是传入值
                        instruct_applied=bool(chunk.instruct))

    def release(self) -> None:
        ...                                    # 卸权重 + empty_cache()
```

### ② 登记到注册表 `core/registry.py`

```python
_ENGINE_MODULES = {..., "mytts": "agentic_tts.engines.mytts"}
```

顺便把名字加进 `engine_specs()` 的遍历列表，`tts engines` 才会列出它。

### ③ 配置里选它

```yaml
engine:
  backend: mytts
  options: {...}        # 引擎私有参数放这里，不必改 EngineConfig 的字段
```

### ④ 写测试

只测这个引擎自己那一层。真实权重的用例标 `@pytest.mark.real`（默认不跑）。
门面/服务/GUI 的测试**不用改** —— 它们走 `sine` 引擎，与新引擎无关。

---

## 2 · 五条硬约束（踩过才写下来的）

| # | 约束 | 为什么 |
|---|---|---|
| 1 | **重依赖只在函数体内 import** | `import agentic_tts` 与查能力表都不该拉起 torch —— GUI 要秒开 |
| 2 | **能力表必须静态可查** | GUI 在开机时置灰开关，不可能为此先载几个 GB |
| 3 | **不支持就 `require()` 报错** | 静默降级是最难查的一类问题（上游 0.6B 静默丢 `instruct` 就是这个形状） |
| 4 | **`release()` 里必须清缓存** | 仅 `del` 不把显存还给驱动，下次加载照样 OOM 且报错看不出原因 |
| 5 | **别在引擎里做分段/插静音/文本预处理** | 见本文开头 |

## 3 · Vocal events 怎么接

- 引擎**有原生事件标记**（Breeze-TTS 2 / IndexTTS-2 宣称有，未实测）：
  声明 `Capability.VOCAL_EVENTS`。门面会把 `[laugh]` 这类标记**原样留在文本里**交给你，
  由你映射成自己的原生写法。
- 引擎**没有**（Qwen3-TTS 就没有）：什么都不用做。门面会把标记删掉、转成中文
  instruct 提示，并在 `warnings` 里写明"效果不保证"。
- `[pause:400ms]` **永远由门面处理**（切段 + 插静音），任何引擎都不必管 ——
  它是唯一确定性的节奏控制手段。

## 4 · 当前占位的两个

`agentic_tts/engines/placeholders.py` 里 `breeze2` / `index2` 只声明能力表（按公开文档，
**一条都没实测**）并在实例化时报「本阶段未接入」。这样 GUI 能提前置灰、配置里写错
能得到一句能照着做的话，而不是 ImportError。真正接入时**必须逐条核对能力表**，
尤其 `VOCAL_EVENTS`。
