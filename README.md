# Agentic TTS Module

可插拔的**本地** TTS 服务模块。默认引擎 Qwen3-TTS 1.7B（PyTorch，cpu/cuda），
支持内置音色 / 音色设计 / 音色克隆 / vocal events；长文本的规范化-分段-拼接-失控兜底
由模块统一负责。库、CLI、HTTP 服务、GUI 四种用法。

> 定位：**输入文本与音色意图，输出高质量音频。** 不做文案生成/改写（那是调用方的 LLM
> 的事，本模块不含任何 LLM），不做音质评测（那是 `verify_qwen3_tts` 的事）。

---

## 快速开始

```bash
conda activate ov_env_py312
cd Agent_TTS_Module

# 1) 体检：配置、权重、设备、**实际导入的 qwen_tts 是哪一份**
python -m agentic_tts.cli doctor

# 2) 合成一句
python -m agentic_tts.cli speak "今天的人工智能资讯。" --voice Serena

# 3) 图形界面（单段试听 + 多段流水线）
python -m agentic_tts.gui

# 4) HTTP 服务
python -m agentic_tts.server            # http://127.0.0.1:8300/docs
```

不想加载权重先试界面/接口：所有命令都支持 `--engine sine`（正弦波占位引擎，零重依赖）。

装成命令（可选）：`pip install -e .` 之后可以直接用 `tts doctor` / `tts speak …`。

---

## 库用法

```python
from agentic_tts import TTSModule, SynthRequest

tts = TTSModule()                                   # 读 configs/config.yaml

# 内置音色
r = tts.synthesize("今天的人工智能资讯。", voice="Serena")
print(r.path, r.seconds, r.rtf)

# 音色设计（换 *-VoiceDesign 权重）
r = tts.synthesize(SynthRequest(text="开始播报。", instruct="沉稳的女声，语速平缓"))

# 音色克隆（换 *-Base 权重）
r = tts.synthesize(SynthRequest(text="开始播报。", ref_audio="voices/refs/a.wav",
                                ref_text="参考音频里的原话"))

# 多段：每段各自的音色/模式/事件，每段单独落盘 + 合并一条
pieces = tts.split("旁白：今天的资讯。\n记者：第一条消息。", rule="role")
r = tts.synthesize_many([SynthRequest(text=p.text, role=p.role, voice=p.voice or "Serena")
                         for p in pieces], merge=True)

tts.release()      # 卸权重、清显存
```

## Vocal events

| 标记 | 行为 | 保证 |
|---|---|---|
| `[pause:400ms]` `[pause]` `[停顿:1s]` | 切段并插入等长静音 | ✅ 确定性，与引擎无关 |
| `[laugh] [sigh] [breath] [cry] [whisper] [cough] [shout] …` | 转成中文 instruct 提示 | ⚠️ 尽力而为，会在 `warnings` 里说明 |

字幕：合并音频时按 `output.subtitles` 顺带写一份 `merged.srt`（一段音频一条字幕，
时间轴含段间静音），剪映 / Premiere / DaVinci 可直接导入。

**Qwen3-TTS 开源权重没有原生事件 token**（tokenizer 特殊符号全表 + 全仓库 grep 双向核实，
见 `docs/00_research.md §3`）。有原生支持的引擎（Breeze-TTS 2 / IndexTTS-2）接入后
会声明 `vocal_events` 能力，届时标记原样交给引擎。

---

## 目录

```
agentic_tts/
  core/          配置、类型、注册表、异常、日志
  engines/       qwen3（默认）、sine（离线自测）、placeholders（breeze2/index2 占位）
  text/          预处理（URL/Markdown/数字）、分段、拆条规则、事件标记
  audio/         读写/拼接/重采样、失控兜底
  voices/        音色档案（内置音色 + 克隆参考）
  engine.py      门面 TTSModule —— 唯一需要记住的类
  server/ gui/ cli.py
configs/config.yaml   voices/voices.yaml   third_party/qwen3_tts（vendored 上游源码）
docs/   tests/   scripts/client.py
```

引擎抽象只有**一个**必实现方法（`synthesize(chunk)`）+ 一张能力表：
文本链路、失控兜底、拼接、轮动排序全部在门面上，对所有引擎共享。
新增引擎见 `docs/06_add_engine.md`。

## 8 GB 显卡（4060）上的轮动加载

实测一个 1.7B checkpoint：bf16 权重 3.9 GB，跑起来约 5 GB —— **装得下一个，装不下两个**。
三个模式各绑一份权重，所以：

- `engine.max_resident: 1`（默认）：换模式时**先卸再载**，并显式 `empty_cache()`；
- 多段混用不同模式时，门面**按 checkpoint 分组执行**，把轮动次数从「段数」压到「模式数」；
- 加载前预检显存，不够就直接报错并给出该改哪一项（不必白等几十秒才 OOM）。

## 测试

```bash
python -m pytest tests -q            # 离线，秒级（126 条，不加载任何权重）
python -m pytest tests -q -m real    # 真实模型，手动触发（每条几十秒起）
```

## 文档

| 文件 | 内容 |
|---|---|
| `docs/00_research.md` | 调研：上游 API、10 条可复用的实测结论、环境体检、事件 token 核查 |
| `docs/01_design.md` | 方案：分层、模式选路、文本链路、兜底、配置、接口 |
| `docs/02_dev_plan.md` | 开发计划与测试策略 |
| `docs/03_unit_tests.md` | 单元测试说明与复测命令 |
| `docs/04_api_reference.md` | 库 / HTTP / CLI 接口 |
| `docs/05_deployment.md` | 本机 CPU、4060 CUDA、服务化部署 |
| `docs/06_add_engine.md` | 怎么接一个新 TTS 引擎 |
| `docs/07_gui_guide.md` | GUI 使用说明 |
| `docs/08_git_commands.md` | 分步提交指令（人工执行） |
| `docs/09_dev_log.md` | 开发与测试记录：实测数字、与方案的偏差 |
| `docs/issues/` | 开发中发现的三个问题（含回归用例） |

项目专属的开发约定作为 Claude Code skill 保存在
`~/.claude/skills/agentic-tts-dev/SKILL.md`（分层规则、8GB 轮动约束、
文本链路顺序、测试策略、提交纪律）。
