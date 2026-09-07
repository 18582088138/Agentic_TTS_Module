# 01 · 方案设计 / Design

> 依据 `00_research.md` 的实测结论 + 2026-09-07 的四条决策：
> ①交付面＝库 + CLI + HTTP + 最简 GUI　②长文本处理放模块内（默认开、可关）
> ③Vocal Events＝停顿确定性 + 其余转 instruct　④**本模块不含任何 LLM 功能**（纯 TTS）

---

## 1 · 一句话定位

> **输入文本与音色意图，输出高质量音频。** 支持多 TTS 引擎（默认 Qwen3-TTS 1.7B，
> torch cpu/cuda），Voice Clone / Voice Design / Vocal Events 通过参数选路，
> 长文本的规范化-分段-拼接-失控兜底由模块统一负责。

不做的事（明确划界）：不做文案生成/改写（那是 LLM 的事，归调用方）、不做 ASR/指标评测
（那是 `verify_qwen3_tts` 的事）、不做音视频混流（归 DNA）。

---

## 2 · 分层：引擎只管「一段文本 → 一段波形」

这是整个设计里最重要的一条边界。

```
                 ┌──────────────────────── TTSModule（门面 engine.py）──────────────────────┐
 SynthRequest →  │ text.prepare → text.segment → events 解析 → [逐段] engine.synthesize    │ → SynthResult
                 │                                 ↑ audio.guard 失控检测/换种子重试        │
                 │                            audio.io 拼接 + 插静音 + 落盘                  │
                 └──────────────────────────────────────────────────────────────────────────┘
                                            ↓ 只此一个抽象方法
                        TTSEngine.synthesize(chunk) -> (wav, sample_rate, effective_gen)
                        qwen3 / sine（离线自测）/ breeze2·index2（占位）
```

**为什么这么切**：文本规范化、分段、静音拼接、失控兜底这四件事**与引擎无关**，
一旦下沉到各引擎实现就会出现 N 份互相偷偷不一致的分段规则（DNA 里 `segment.py` 与
textkit 双份并存正是这个坑）。放在门面上，新增一个引擎只需实现一个方法。

引擎侧只声明**能力**，门面按能力选路或明确拒绝：

```python
class Capability(str, Enum):
    CUSTOM_VOICE      # 内置音色 + speaker id
    VOICE_DESIGN      # 自然语言描述音色
    VOICE_CLONE       # 参考音频克隆（x-vector）
    CLONE_ICL         # 克隆的 ICL 模式（要 ref_text，音色更像）
    INSTRUCT          # 指令控制语气/语速
    VOCAL_EVENTS      # 原生事件标记（Qwen3 **没有**，见 00_research §3）
    BATCH             # 一次调用多条文本
```

| 引擎 | 能力 | 本阶段 |
|---|---|---|
| `qwen3` | CUSTOM_VOICE, VOICE_DESIGN, VOICE_CLONE, CLONE_ICL, INSTRUCT, BATCH | ✅ 实现并实测 |
| `sine` | 无（只产正弦波） | ✅ 离线自测用，零依赖 |
| `breeze2` / `index2` | 待定 | 只在注册表留名，实例化时报清晰错误 |

`PAUSE`（插静音）不在能力表里 —— 它由门面在拼接层实现，**对所有引擎都成立**。

---

## 3 · 模式选路：三 API 各绑一个 checkpoint

上游门禁是硬写的（`00_research §1`），所以 mode 决定 checkpoint，**换 mode = 换权重**。

```
mode 推断（未显式指定时）：
  ref_audio 有值            → voice_clone   → *-Base
  voice 为空 且 instruct 有值 → voice_design  → *-VoiceDesign
  其他                      → custom_voice  → *-CustomVoice（默认 voice=Serena）
```

推断规则写死在一个函数里并有单元测试，`mode=` 可显式覆盖。
参数与 mode 冲突时**直接报错**（例：`mode=custom_voice` 却给了 `ref_audio`），
不做「悄悄忽略」—— 上游 0.6B 静默丢弃 instruct 那个坑的形状就是这样。

**模型池 `max_resident` 默认 1**（8 GB 卡装不下两个 1.7B，实测），换 checkpoint 先卸再载，
`torch.cuda.empty_cache()` 必调。跨 mode 的工作流（design→clone）**分两阶段跑**，
不靠提高上限。

---

## 4 · 文本链路

```
prepare(text) ──► events.parse ──► segment ──► 逐段合成
  URL → 读法/丢弃      [pause:400ms] → 静音        按句读点切
  Markdown 残留清理    [laugh] 等   → instruct 提示  上限 max_chars
  数字/单位/日期规范化
```

- **移植范围收着来**：`verify_qwen3_tts/src/vq3/textkit/` 是纯函数零依赖，
  但实测只有 **URL（CER 4.43→0）与 Markdown（0.83→0.17）** 收益大；
  小数/emoji 处理了白做 → 精简移植，**多音字/专名词典默认关闭**（它会换字，
  返回的文本只能送 TTS 不能显示给人）。
- 每个开关都能从 config 关掉，`prepare_text=False` 时走裸合成，便于 A/B。
- 分段的第四条理由（实测）：失控只毁掉一段，可单独重试。

### Vocal Events 语义（决策③）

| 标记 | 行为 | 保证 |
|---|---|---|
| `[pause:400ms]` / `[pause]` | 切段，在拼接处插静音 | ✅ 确定性 |
| `[laugh] [sigh] [breath] [cry] [whisper]` | 从标记生成中文 instruct 提示追加到该段的 instruct 后；文本里的标记**删除** | ⚠️ 尽力而为 |
| 引擎声明 `VOCAL_EVENTS` 时 | 交给引擎映射成原生标记，门面不改写 | 由引擎保证 |

`SynthResult.warnings` 会明确写「本引擎无原生事件支持，已转 instruct，效果不保证」——
**能力不足要说出来，不能让调用方以为生效了。**

---

## 5 · 失控兜底（`audio/guard.py`）

实测：模型会停不下来，产出几十秒噪声且**不报错**，波形指标与静音检测都测不出来。

```python
cap_seconds = max_new_tokens / codes_per_second   # Qwen3 12Hz → /12.5
runaway = seconds >= cap_seconds * 0.95           # 主判据：撞上限
          or seconds > len(text) / chars_per_sec * ratio   # 辅：中文 ~5 字/秒，ratio 默认 3
→ 换种子重试 retry 次（默认 1）；仍失败：该段标记 failed，不把噪声混进成品
```

**不关采样**：实测贪心解码 12 条里 7 条失控、官方采样默认值 0 条失控。
默认一个生成参数都不传（走 checkpoint 的 `generation_config.json`），
`gen={}` 里的覆盖项与「实际生效值」一起记进结果（调 `_merge_generate_kwargs` 拿真值）。

---

## 6 · 音色档案（`voices/`）

一层 YAML，把「内置 speaker」与「克隆参考音频」统一成一个名字：

```yaml
# voices/voices.yaml
serena:   { engine: qwen3, mode: custom_voice, speaker: Serena, language: Chinese }
anchor_f: { engine: qwen3, mode: voice_clone, ref_audio: refs/anchor_f.wav,
            ref_text: "今天的人工智能资讯", x_vector_only: false }
calm_f:   { engine: qwen3, mode: voice_design,
            instruct: "沉稳的女声，语速平缓，播音腔" }
```

调用方只需 `voice="anchor_f"`，不必知道背后是哪套 API。参考音频路径相对 `voices/` 解析
（**所有路径统一相对一个根**，避免「搬了一半」）。`ref_audio` 只支持本地文件与 base64 ——
上游的 URL 加载不读代理配置，公司网络下会超时（实测）。

---

## 7 · 配置：单一 YAML + 环境变量覆盖

`configs/config.yaml` 是唯一真源，`TTS_*` 环境变量可覆盖（部署时不改文件）。
checkpoint **配目录名、统一相对 `models_dir` 解析**。

```yaml
engine:
  backend: qwen3            # qwen3 | sine | breeze2(占位) | index2(占位)
  device: cpu               # cpu | cuda | cuda:0 | auto
  dtype: auto               # auto(cpu→float32, cuda→bfloat16) | bfloat16 | float16 | float32
  attn_impl: eager
  seed: 20260907
  max_resident: 1
  models_dir: "C:/Users/test/Downloads/xkd/Models"
  repo_dir:  "C:/Users/test/Downloads/xkd/verify_qwen3_tts/Qwen3-TTS"   # 见下
  checkpoints:
    custom_voice: Qwen3-TTS-12Hz-1.7B-CustomVoice
    voice_design: Qwen3-TTS-12Hz-1.7B-VoiceDesign
    base:         Qwen3-TTS-12Hz-1.7B-Base
text:   { prepare: true, urls: true, markdown: true, numbers: true, lexicon: false,
          segment: true, max_chars: 80 }
audio:  { sample_rate: null, format: wav, default_pause_ms: 300, join_silence_ms: 120 }
guard:  { enabled: true, retry: 1, chars_per_sec: 5.0, duration_ratio: 3.0 }
voices: { dir: "./voices" }
server: { host: 127.0.0.1, port: 8300 }
output: { dir: "./outputs" }
```

⚠️ **`repo_dir` 必须显式配**：环境里 `qwen-tts 0.0.4` 的 editable 安装是坏的，
且磁盘上两份源码 commit 不同。本模块把 `repo_dir` 插到 `sys.path[0]`，**不动全局安装**
（同一 conda 环境 DNA 也在用），并在 `doctor` 里打印实际导入的 `qwen_tts.__file__`。

---

## 8 · 目录结构

```
Agent_TTS_Module/
  agentic_tts/
    __init__.py            # 只导出 TTSModule 与版本，**不 import torch**
    core/{config,types,registry,errors,logging}.py
    engines/{base,qwen3,sine}.py          # 注册表懒加载：选中才 import 重依赖
    text/{__init__,normalize,segment,events}.py
    audio/{io,guard}.py
    voices/store.py
    engine.py                             # TTSModule 门面
    server/{app.py,__main__.py}           # FastAPI
    gui/{app.py,__main__.py}              # NiceGUI（3.14 已装）
    cli.py                                # typer: doctor/speak/clone/design/voices/serve
  configs/config.yaml   voices/   scripts/client.py
  docs/   tests/   outputs/（gitignore）
```

GUI 与 HTTP 服务是**并列**的两种用法：GUI 同进程直调 `TTSModule`，不经 HTTP
（抄 RAG 的结论，单机用户不必先起 server）。
⚠️ NiceGUI 的 UI 必须建在 `@ui.page('/')` 里 —— 顶层建会让入口脚本被重跑并 500（RAG 实测）。

## 9 · 对外接口（门面 = HTTP = CLI 一一对应）

```python
TTSModule(config=None)
  .synthesize(req: SynthRequest | str, **kw) -> SynthResult   # 主入口
  .synthesize_to_file(req, path) -> Path
  .voices() -> list[VoiceInfo]        # 内置 speaker + 档案
  .languages() -> list[str]
  .capabilities() -> set[Capability]
  .info() -> dict                     # 引擎/设备/dtype/checkpoint/版本快照
  .release() -> None                  # 卸模型、清显存
```

| HTTP | 对应 |
|---|---|
| `GET /health` `GET /info` `GET /voices` `GET /capabilities` | 同名方法 |
| `POST /tts/synthesize` | `synthesize`，返回 wav 二进制或 base64（按 `encoding`） |
| `POST /tts/clone` `POST /tts/design` | 语法糖，内部固定 mode |

`SynthResult` 必带的可观测字段：`seconds / rtf / sample_rate / segments[] /
effective_gen / instruct_applied / warnings[] / engine_info` ——
「这条音频到底是用什么参数跑出来的」是事后最想知道的事（verify 项目的教训）。
